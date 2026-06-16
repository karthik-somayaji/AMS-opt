import numpy as np
import re
import sys
import json
import os
import random

sys.path.append("../")
from core.fewshot_agent import FewShotAgent # noqa: E402
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


class LLMProposer(FewShotAgent):
    def __init__(self,
                 ckt_name_description,
                 params_list,
                 task_context,
                 backend,
                 example_keys,
                 n_proposal=1,
                 ranges=None,
                 max_request_attempt=3,
                 ):
        super().__init__(
            example_keys=example_keys
        )
        self.ckt_name_description = ckt_name_description
        self.task_context = task_context
        self.params_list = params_list
        self.params_units = ["f", "m", "n", "p", "u", "k", "G", "M"]
        self.n_params = len(self.params_list)

        self.backend = backend
        self.backend_history = backend
        self.max_request_attempt = max_request_attempt  # TODO: change request to adapt to larger number of proposal;

        self.ranges = ranges
        self.n_proposal = n_proposal

        self.debug_mode = False

        self.num_calls = 0

    def propose_params(self, data_collected, args):
        if self.n_proposal == 0:
            return []
        else:
            examples = self.generate_examples(data_collected)

            # Propose candidate points within maximum attempts;
            #print(self.max_request_attempt, self.n_proposal)
            params_proposed = []
            for _ in range(self.max_request_attempt):
                #print('EXAMPLES: ', examples)
                prompt = self.generate_prompt(examples, "", args)
                responses = self.backend.request(prompt)
                # print(responses)
                params_parsed = self.parse_llm_responses(responses)

                params_proposed += params_parsed

                if self.debug_mode:
                    print("***Prompt***\n")
                    #print(f"{prompt}")

                    print("***Response***\n")
                    #print(f"{responses[0]}")

                if len(params_proposed) >= self.n_proposal:
                    break

            if len(params_proposed) < self.n_proposal:
                raise ValueError("LLM failed to propose enough valid data!!")
            else:
                params_proposed = params_proposed[0:self.n_proposal]
                self.num_calls += 1
                return params_proposed


    def generate_prefix(self, prompt_input, args):
        # TODO: looks like the GPT seems to be lazy and did not think to much for the given task context and sequence of the prompt_input;
        
        prefix = """ """
        # prefix = self.task_context

        model = args.model

        # related circuits configuration (optional CLI args)
        related_mode = getattr(args, "related_mode", "topk")
        related_k = int(getattr(args, "related_k", 3))
        embeddings_json = getattr(args, "embeddings_json", "umap_results_full/umap_embeddings.json")

        # Map task circuit description/name -> netlists family
        family_map = {
            "amp2": "diff_amps",
            "FC": "diff_amps",
            "ldo": "LDO",
            "comp": "comparators",
        }

        ado_kt_map = {
            "amp2": ["FC"],
            "FC": ["amp2"],
            "comp": ["amp2"],
            "ldo": ["amp2", "comp"],
        }

        # IMPORTANT: `netlists/<family>/<id>` is used only to load related-circuit KG context
        # (fun_graph.json) for prompting. Optimization/simulation targets come from each task's
        # `ckt_dir` (e.g., `LLMBO/*_ati_new/`).

        # related_ckts = [ 'Two_Stage_Differential_Amplifier']
        # related_ckts = [  'Hysteresis_Comparator', 'Two_Stage_Differential_Amplifier']
        #related_ckts = [ 'Low_Dropout_Regulator']
        #related_ckts =  [ 'Hysteresis_Comparator',   'Low_Dropout_Regulator', 'Two_Stage_Differential_Amplifier']#, 'Low_Dropout_Regulator']#['Two_Stage_Differential_Amplifier', 'Hysteresis_Comparator']#['Two_Stage_Differential_Amplifier']##, 'Folded_Cascode_Amplifier', 'Hysteresis_Comparator']

        def _repo_root() -> str:
            # `proposer.py` lives in `LLMBO/core/`
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        def _load_embeddings(path: str):
            path_abs = path if os.path.isabs(path) else os.path.join(_repo_root(), path)
            with open(path_abs, "r") as f:
                data = json.load(f)
            embeds = np.array(data["embeddings"], dtype=float)
            meta = data["metadata"]
            return embeds, meta

        def _pick_k_similar(family: str, target_id: str, k: int, *, descending: bool = True):
            embeds, meta = _load_embeddings(embeddings_json)
            idxs = [i for i, m in enumerate(meta) if m.get("family") == family]
            id_to_i = {meta[i]["circuit_id"]: i for i in idxs}
            anchor_vec = None
            if target_id in id_to_i:
                anchor_vec = embeds[id_to_i[target_id]]
            else:
                # Allow LLMBO anchor embeddings computed from `LLMBO/<id>_ati_new/comb_graph_gnn.npz`.
                # See `LLMBO/llmbo.py` which sets `args.target_anchor_embedding` for symbolic target ids.
                maybe_anchor = getattr(args, "target_anchor_embedding", None)
                if maybe_anchor is not None:
                    anchor_vec = np.array(maybe_anchor, dtype=float)
                    if anchor_vec.ndim != 1:
                        anchor_vec = anchor_vec.reshape(-1)

            if anchor_vec is None:
                available = sorted({str(meta[i].get("circuit_id")) for i in idxs})
                sample = ", ".join(available[:20])
                print(
                    "[LLMProposer] WARNING: --target_id not found in embeddings for this family and no "
                    "target_anchor_embedding provided; falling back to random same-family KGs. "
                    f"family={family} target_id={target_id}. Example valid ids: {sample}"
                )
                return _pick_random_same_family(family, target_id, k)

            if int(anchor_vec.shape[0]) != int(embeds.shape[1]):
                raise ValueError(
                    "Anchor embedding dimension does not match reference embeddings. "
                    f"Got anchor_dim={int(anchor_vec.shape[0])} but reference_dim={int(embeds.shape[1])}. "
                    "Make sure --embeddings_json points to RAW GNN embeddings (not 2D UMAP)."
                )

            target_vec = anchor_vec
            vecs = embeds[idxs]
            den = (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(target_vec) + 1e-8) + 1e-8)
            sims = (vecs @ target_vec) / den
            pairs = [(idxs[j], float(sims[j])) for j in range(len(idxs)) if meta[idxs[j]]["circuit_id"] != target_id]
            pairs.sort(key=lambda x: x[1], reverse=bool(descending))
            return [(meta[i]["circuit_id"], meta[i]["family"], s) for i, s in pairs[:k]]

        def _pick_random_same_family(family: str, target_id: str, k: int):
            family_dir = os.path.join(_repo_root(), "netlists", family)
            ids = []
            if os.path.isdir(family_dir):
                for name in os.listdir(family_dir):
                    if os.path.isdir(os.path.join(family_dir, name)):
                        ids.append(name)
            ids = [x for x in ids if x != str(target_id)]
            if len(ids) == 0:
                return []
            random.shuffle(ids)
            return [(cid, family, None) for cid in ids[:k]]

        def _load_fun_graph_text(family: str, circuit_id: str) -> str:
            path = os.path.join(_repo_root(), "netlists", family, str(circuit_id), "fun_graph.json")
            return _load_fun_graph_text_from_path(path)

        def _load_fun_graph_text_from_path(path: str) -> str:
            with open(path, "r") as f:
                # LangChain PromptTemplate uses `{...}` for variables; escape braces in raw JSON.
                txt = f.read()
                txt = txt.replace("{", "{{").replace("}", "}}")
                return txt

        def _infer_target_circuit_key() -> str | None:
            circuit = getattr(args, "circuit", None)
            if circuit in ado_kt_map:
                return str(circuit)

            target_id = getattr(args, "target_id", None)
            if target_id in ado_kt_map:
                return str(target_id)

            desc = str(self.ckt_name_description or "")
            desc_lower = desc.lower()
            if "folded" in desc_lower and "cascode" in desc_lower:
                return "FC"
            if "hysteresis" in desc_lower and "comparator" in desc_lower:
                return "comp"
            if "dropout" in desc_lower or "ldo" in desc_lower:
                return "ldo"
            if "two-stage" in desc_lower or "two stage" in desc_lower:
                return "amp2"
            return None

        def _pick_ado_kt_related(target_circuit: str):
            if target_circuit not in ado_kt_map:
                raise ValueError(
                    "related_mode='ado-kt' is supported only for amp2, FC, comp, and ldo. "
                    f"Got target circuit: {target_circuit!r}."
                )

            related = []
            for circuit_name in ado_kt_map[target_circuit]:
                kg_path = os.path.join(_repo_root(), "LLMBO", f"{circuit_name}_ati_new", "fun_graph.json")
                related.append({"circuit": circuit_name, "kg_path": kg_path})
            return related

        def _infer_target_family_and_id():
            target_family = family_map.get(self.ckt_name_description, None)
            # If task name isn't one of the keys above, fall back to the JSON's ckt name.
            if target_family is None:
                # heuristic based on description
                desc = (self.ckt_name_description or "").lower()
                if "comparator" in desc:
                    target_family = "comparators"
                elif "dropout" in desc or "ldo" in desc:
                    target_family = "LDO"
                else:
                    target_family = "diff_amps"

            # circuit id must come from CLI (e.g., --target_id 77)
            target_id = getattr(args, "target_id", None)
            if target_id is None:
                raise ValueError("Missing --target_id (required to select related circuits by similarity).")
            return target_family, str(target_id)

        multiline_history = ""
        if args.history:
            if related_mode == "ado-kt":
                target_circuit = _infer_target_circuit_key()
                if target_circuit is None:
                    raise ValueError(
                        "Unable to infer the optimization circuit for related_mode='ado-kt'. "
                        "Pass --circuit explicitly."
                    )
                related = _pick_ado_kt_related(target_circuit)
                for item in related:
                    try:
                        kg_json = _load_fun_graph_text_from_path(item["kg_path"])
                        header = f"\n\n[RELATED_CIRCUIT circuit={item['circuit']} source=ado-kt]\n"
                        multiline_history += header + kg_json
                    except FileNotFoundError:
                        continue
            else:
                target_family, target_id = _infer_target_family_and_id()
                if related_mode == "random_family":
                    related = _pick_random_same_family(target_family, target_id, related_k)
                elif related_mode == "bottomk":
                    related = _pick_k_similar(target_family, target_id, related_k, descending=False)
                else:
                    related = _pick_k_similar(target_family, target_id, related_k, descending=True)

                for rid, rfamily, score in related:
                    try:
                        kg_json = _load_fun_graph_text(rfamily, rid)
                        header = f"\n\n[RELATED_CIRCUIT family={rfamily} id={rid}"
                        if score is not None:
                            header += f" similarity={score:.4f}"
                        header += "]\n"
                        multiline_history += header + kg_json
                    except FileNotFoundError:
                        continue

        # Include related-circuit KG context if enabled
        if args.history and multiline_history:
            prefix += multiline_history

        prefix += self.task_context

        # prefix += f"You need to optimize the size of these parameters: {self.params_list} to achieve optimal target value. "
        # prefix += "The target value is defined as the gain of the circuit. "
        # prefix += "**Examples** gives demonstration of existing parameter settings and the simulated target values from HSPICE. "
        # prefix += f"Based on the netlist and demonstrations, summarize how you can adjust the sizing of parameters to achievea target of {prompt_input}. "
        # prefix += "Then propose a different set of parameters in the format of params in **Examples**. "
        # prefix += "You must not add any comments beyond the recommendation.\n"
        prefix += "**Examples**\n"
        # prefix += ""

        #print("PREFIXXXXXXXXXXXXXXXXXXXXX: ", prefix)

        return prefix

    def generate_suffix(self, prompt_input):
        suffix = """Based on the netlist and demonstrations and the design rules of the two stage differential amplifier circuit, contemplate on how you can adjust the sizing of parameters to achieve the desired specifications.
        Specifically, use common circuit sub-structures between the two structures and then use the sizing rules mentioned for the matching sub-scircuit in the two stage differential amplifier to size different sub-circuits of the folded cascode amplifier.
        Then propose a different set of parameters (all in the range [0,1]) in the format of params in **Examples**. You must not add any comments beyond the recommendation.
        """

        suffix = """ """

        #suffix += f"Note that your design should be an exact copy of the BEST parameter settings (examples). Also note the parameters proposed should be in the format of params in **Examples**. Please provide only the paramaters. No explanation required.\n "
        ##suffix += f"Note that your design should be neither too far from the BEST parameter settings (examples), nor too close (or same) to the BEST parameter settings (examples). Also note the parameters proposed should be in the format of params in **Examples**. Please provide only the paramaters. No explanation required.\n "
        #suffix += f"Note that your suggested design should be neither too far from the BEST parameter settings (examples), nor too close to the BEST parameter (can be maximum of 0.2 away from the BEST parameter). Your response should only include the parameters which should be in the format of params in **Examples** and should strictly be between 0 and 1 only and with a precision of 2 atleast!. Your answer should be in the format of params in **Examples** \n "
        suffix += f"Note that your suggested design should be far from the BEST parameter settings (examples) in accordance with the design knowledge for this circuit. Your response should only include the parameters which should be in the format of params in **Examples** and should strictly be between 0 and 1 only and with a precision of 2 atleast!. Your answer should be in the format of params in **Examples** \n "
        #suffix = f"The suggested parameters should be an exact copy of the parameters corresponding to the highest target value from **examples**. Note the paameters proposed should be in the format of params in **Examples**\n "
        #suffix = f"The suggested parameters should be such that all transistors will be in saturation region. Note the parameters proposed should be in the format of params in **Examples**\n "
        return suffix
    
    def generate_prompt_history_summary(self, data_collected):

        '''
        Takes as input, data collected (given as argument) and the 
        prefix to gnerate some design rules specific to different
        sub-compenents of the circuit. 

        '''
        examples = self.generate_examples_for_history(data_collected)
        prefix = self.generate_prefix_for_history()
        suffix = self.generate_suffix_for_history()

        #print('EXAMPLES FOR HISTORY : ', len(examples), examples)

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=[""],
            example_separator="\n"
        )

        prompt_for_history = few_shot_prompt.format(prompt_input="")#.format(prompt_input=f"{prompt_input}")
        # responses_for_history = self.backend_history.request(prompt_for_history)

        #print(prompt_for_history)

        #print('*********************')

        #print(responses_for_history)

        # with open('/home/karthik/sim_clean/LLMBO/history_summary/' + self.ckt_name_description +'.txt', 'w') as file:
        #     file.write(responses_for_history[0])

        with open('/home/karthik/sim_clean/LLMBO/history_summary/prompt_' + self.ckt_name_description +'.txt', 'w') as file:
            file.write(prompt_for_history)
        
        return [] #responses_for_history


    def generate_prefix_for_history(self):
        #prefix = "You are an analog circuit designer designing sizing solutions for multiple circuits. You will use the design knowldge of previous circuits to improve upn the design of the current circuit.\n "
        prefix = "This is the two-stage differenial amplifier circuit. Its description is mentioned below. \n"
        prefix += """ **Circuit Netlist** gives an HSPICE netlist of a two-stage differential amplifier.\n:
        **Circuit Netlist**
.GLOBAL vdd!

.TEMP 25
.OPTION
+    ARTIST=2
+    INGOLD=2
+    MEASOUT=1
+    PARHIER=LOCAL
+    PSF=2
+	 OPFILE=1

cload out 0 1e-11
v_sup vdd! 0 DC=2
vcm vcm 0 DC=1 AC=vacc
vin vac 0 DC=0 AC=vacd
e1 vip vcm vac 0 0.5
e2 vin vcm 0 vac 0.5
ib vdd! vb 5e-6

xp1 node1 node1 vdd! vdd! pfet l='80e-9+l1*920e-9' w='120e-9+w1*49880e-9'
xp2 out1 node1 vdd! vdd!  pfet l='80e-9+l2*920e-9' w='120e-9+w2*49880e-9'
xn1 node1 vip node2 0     nfet l='80e-9+l3*920e-9' w='120e-9+w3*49880e-9'
xn2 out1 vin node2 0      nfet l='80e-9+l4*920e-9' w='120e-9+w4*49880e-9'
xnb vb vb 0 0             nfet l='80e-9+l5*920e-9' w='120e-9+w5*49880e-9'
xnc node2 vb 0 0          nfet l='80e-9+l6*920e-9' w='120e-9+w6*49880e-9'
xpo out out1 vdd! vdd!    pfet l='80e-9+l7*920e-9' w='120e-9+w7*49880e-9'
xno out vb 0 0            nfet l='80e-9+l8*920e-9' w='120e-9+w8*49880e-9'
cc out outm 'pwr(10,c1*4-2)*1e-12'
rz outm out1 'pwr(10,r1*4-2)*1e3'

[l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, l6, w6, l7, w7, l8, w8, r1, c1] need to be tuned to achieve the desired performance on the following metrics: gain: >60 dB, Common-Mode Rejection Ratio (cmrr): >80 dB, Gain Bandwidth Product (gbw): >1 MHz, Phase Margin (pm): >=45 degree.
    **Examples** gives demonstrations of the top-5 existing parameter settings and the simulated metrics from HSPICE.
    """
        with open(f'/home/karthik/sim_clean/LLMBO/history_summary/prefix_{self.ckt_name_description}.txt', 'r') as file:
            prefix = file.read()
        return prefix
        
    def generate_suffix_for_history(self):
        #suffix = """Given the netlist, the transistors, and the top-5 optimal points in terms of .params describe some design rules for each of the sub-circuits for example like differential pair,
        #current mirror transistors, active load transistors, the bias transistors etc. Make sure your respons is no more than 10 lines and in a single paragraph. """

        suffix = """Given the netlist, the transistors, and the top-5 normalized optimal points (in range 0 to 1) in terms of .params describe some design rules for each of the sub-circuits (for example like differential pair,
        current mirror transistors, active load transistors, the bias transistors etc). Make sure your response is organized in the form of points. Define how providing high or low values for the width to length ratio of transistors in the 
        each sub-circuit affects each of the metrics namely the gain, cmrr, gbw and the phase margin. Mention specifically if it increases or decreases the metric. If it is complex, mention so. Mention if there are any trade-offs in the metrics for a certain directon of tuning the widths to lengths ratio. """

        suffix = """Given the netlist, the transistors, and the top-5 normalized optimal points (in range 0 to 1) in terms of .params, describe some design rules for each of the sub-circuits (for example for like differential pair,
        current mirror transistors, active load transistors, the bias transistors etc). Make sure your response is quantitative (as the examples consider normalized values for sizing). Define how providing high or low values for the width to length ratio of transistors in the 
        each sub-circuit affects each of the metrics namely the gain, cmrr, gbw and the phase margin. Generate responses of the form - like increasing sizing the transistor w_1/l_1 = 0.5/0.1 leads to high gain of around 40 dB etc. Make your responses concise, very quantitative with respect to the current circuit and precise. """

        suffix = """Given the netlist, the transistors, and the top-5 normalized optimal points (in range 0 to 1) in terms of .params, describe some design rules for each of the sub-circuits (for example for like differential pair,
        current mirror transistors, active load transistors, the bias transistors etc) in the following format.
        Sub-circuit <structure 1> (composed of transistors <T1, T2 etc>) affects metrics like <Metric 1, etc>. Specifically <increasing width to 0.9 and decreasing length to 0.1 of T1 increases metric 1>.
        Optionally : Providing similar values to <w1, w2 and l1,l2 etc...> has the benefit of boosting <some characteristics>.
        Make sure your response is quantitative (Understand correlations in parameters using the examples and use your prior knowledge on the class of circuit like how a human does). 
        Make your responses concise, very quantitative (needs to have normalized numbers) with respect to the current circuit and precise."""

        with open(f'/home/karthik/sim_clean/LLMBO/history_summary/suffix_{self.ckt_name_description}.txt', 'r') as file:
            suffix = file.read()
        return suffix        

    def parse_llm_responses(self, responses):
        # TODO: range of the params is not checked currently;
        # TODO: add support to log the LLM reasoning part;
        params_units = self.params_units
        params_list = self.params_list

        # Extract formatted sub-strings with the format: param=value_param with regular expression;
        # value_param needs to be float number (optionally) followed by one of the units;
        def extract_params(response):
            found_values = {}
            # Create a regular expression pattern for the units, allowing only specified units;
            # Escape units to avoid regex issues;
            units_pattern = f"[{''.join(re.escape(unit) for unit in params_units)}]?"

            for param in params_list:
                # Regex to find the parameter followed by a number (including decimal numbers) and optionally a unit, allowing spaces;
                match = re.search(rf"{re.escape(param)}\s*=\s*(\d+(\.\d+)?)\b", response)
                if match:
                    # Store the found value using the parameter as the key;
                    found_values[param] = match.group(1)
                else:
                    # If any parameter is not found or its value is invalid, return False;
                    return False

            # formatted_output = "".join(f"{param}={value}" for param, value in found_values.items())
            formatted_output = found_values

            return formatted_output
        

        # Extract for every collected response;
        params_parsed = []
        # Response is None means api call has failed in request attempts (default to be 3);
        if responses is None:
            print(f"Fewshot agent fails to get response from the LLM API after {self.max_request_attempt} attempts!")
            print(f"Api call fails!")
        else:
            for response in responses:
                response_parsed = extract_params(response)
                if response_parsed:
                    params_parsed.append(response_parsed)

        return params_parsed
    
KG_prefix = {
    'Two_Stage_Differential_Amplifier_1': """
     Given the design rules in knowledge graph format for the two stage differential amplifier, use it as PRIOR KNOWLEDGE to optimize the next circuit you are prompted for. Below is the knowledge graph for the two stage differential amplifier.
    """,
    'Two_Stage_Differential_Amplifier': """
You are an analog designer. Given the netlist and design rules of the two stage differential amplifier, use it as PRIOR KNOWLEDGE to optimize other circuits.
**Circuit Netlist of two stage differential amplifier**
.GLOBAL vdd!

.TEMP 25
.OPTION
+    ARTIST=2
+    INGOLD=2
+    MEASOUT=1
+    PARHIER=LOCAL
+    PSF=2
+	 OPFILE=1

cload out 0 1e-11
v_sup vdd! 0 DC=2
vcm vcm 0 DC=1 AC=vacc
vin vac 0 DC=0 AC=vacd
e1 vip vcm vac 0 0.5
e2 vin vcm 0 vac 0.5
ib vdd! vb 5e-6

xp1 node1 node1 vdd! vdd! pfet l=l1 w=w1
xp2 out1 node1 vdd! vdd!  pfet l=l2 w=w2
xn1 node1 vip node2 0     nfet l=l3 w=w3
xn2 out1 vin node2 0      nfet l=l4 w=w4
xnb vb vb 0 0             nfet l=l5 w=w5
xnc node2 vb 0 0          nfet l=l6 w=w6
xpo out out1 vdd! vdd!    pfet l=l7 w=w7
xno out vb 0 0            nfet l=l8 w=w8
cc out outm 'c1'
rz outm out1 'r1'

Design Rules:
""",
"Hysteresis_Comparator": """ Given the design rules in knowledge graph format for the Hysteresis_Comparator, use it as PRIOR KNOWLEDGE to optimize the next circuit you are prompted for. Below is the knowledge graph for the Hysteresis_Comparator.
""",
"Low_Dropout_Regulator": """ Given the design rules in knowledge graph format for the low dropout regulator, use it as PRIOR KNOWLEDGE to optimize the next circuit you are prompted for. Below is the knowledge graph for the low dropout regulator.
""",
"Hysteresis_Comparator_1": """
You are an analog designer. Given the netlist and design rules of the hysteresis comparator, use it as PRIOR KNOWLEDGE to optimize other circuits.
**Circuit Netlist of hysteresis comparator**
cload out 0 1e-11
v_sup vdd! 0 DC=2
vcm vcm 0 DC=1 AC=vacc
vin vac 0 DC=0 AC=vacd PWL(0,-2,10u,2,20u,-2)
e1 vip vcm vac 0 0.5
e2 vin vcm 0 vac 0.5
ib vdd! vb 5e-6

xnb vb vb 0 0       nfet l='80e-9+l11*920e-9' w='120e-9+w11*49880e-9'

xn5 3 vb 0 0        nfet l='80e-9+l12*920e-9' w='120e-9+w12*49880e-9'
xn1 1 vip 3 0       nfet l='80e-9+l1*920e-9' w='120e-9+w1*49880e-9'
xn2 2 vin 3 0       nfet l='80e-9+l2*920e-9' w='120e-9+w2*49880e-9'
xp3 1 1 vdd! vdd!   pfet l='80e-9+l3*920e-9' w='120e-9+w3*49880e-9'
xp4 2 2 vdd! vdd!   pfet l='80e-9+l4*920e-9' w='120e-9+w4*49880e-9'
xp6 2 1 vdd! vdd!   pfet l='80e-9+l5*920e-9' w='120e-9+w5*49880e-9'
xp7 1 2 vdd! vdd!   pfet l='80e-9+l6*920e-9' w='120e-9+w6*49880e-9'

xp8 out 2 vdd! vdd! pfet l='80e-9+l7*920e-9' w='120e-9+w7*49880e-9'
xp9 4 1 vdd! vdd!   pfet l='80e-9+l8*920e-9' w='120e-9+w8*49880e-9'
xn10 4 4 0 0        nfet l='80e-9+l9*920e-9' w='120e-9+w9*49880e-9'
xn11 out 4 0 0      nfet l='80e-9+l10*920e-9' w='120e-9+w10*49880e-9'
"""
}


if __name__ == "__main__":

    from backend.llm.gpt import GPT

    task_context = """
You are an analog circuit design expert. **Circuit Netlist** gives an HSPICE netlist of a two-stage differential amplifier.
You need to optimize the size of these parameters: [l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, l6, w6, l7, w7, l8, w8, r1, c1] to achieve the desired performance on the following metrics:
gain: >60 dB, Common-Mode Rejection Ratio (cmrr): >80 dB, Gain Bandwidth Product (gbw): >1 MHz, Phase Margin (pm): >=45 degree.
**Examples** gives demonstration of existing parameter settings and the simulated metrics from HSPICE.
Based on the netlist and demonstrations, explain how you can adjust the sizing of parameters to achieve the desired specifications.
Then propose a different set of parameters in the format of params in **Examples**.
You must not add any comments beyond the recommendation.

**Circuit Netlist**
.GLOBAL vdd!

.TEMP 25
.OPTION
+    ARTIST=2
+    INGOLD=2
+    MEASOUT=1
+    PARHIER=LOCAL
+    PSF=2
+	 OPFILE=1

cload out 0 1e-11
v_sup vdd! 0 DC=2
vcm vcm 0 DC=1 AC=vacc
vin vac 0 DC=0 AC=vacd
e1 vip vcm vac 0 0.5
e2 vin vcm 0 vac 0.5
ib vdd! vb 5e-6

xp1 node1 node1 vdd! vdd! pfet l=l1 w=w1
xp2 out1 node1 vdd! vdd!  pfet l=l2 w=w2
xn1 node1 vip node2 0     nfet l=l3 w=w3
xn2 out1 vin node2 0      nfet l=l4 w=w4
xnb vb vb 0 0             nfet l=l5 w=w5
xnc node2 vb 0 0          nfet l=l6 w=w6
xpo out out1 vdd! vdd!    pfet l=l7 w=w7
xno out vb 0 0            nfet l=l8 w=w8
cc out outm 'c1'
rz outm out1 'r1'
    """

    # Test of the prompts generated by the proposer;
    b = GPT(model="3.5")
    p = LLMProposer(
        params_list=["w1", "l1", "w2", "l2", "w3", "l3", "w4", "l4", "w5", "l5", "w6", "l6", "w7", "l7", "w8", "l8", "c1", "r1"],
        n_proposal=5,
        task_context=task_context,
        backend=b,
        # example_keys=["params", "metrics", "targets", "aux_info"]
        example_keys=["params", "metrics"]
    )
    data_collected = {
        "params": [
            {
              'w1': '150u', 'l1': '1u', 'w2': '150u', 'l2': '1u', 'w3': '15u', 'l3': '1u', 'w4': '15u',
              'l4': '1u', 'w5': '7u', 'l5': '1u', 'w6': '7u', 'l6': '1u', 'w7': '150u', 'l8': '1u',
              'c1': '1p', 'r1': '1k'
            },
            {
                'w1': '150u', 'l1': '1u', 'w2': '150u', 'l2': '1u', 'w3': '15u', 'l3': '1u',
                'w4': '15u', 'l4': '1u', 'w5': '7.5u', 'l5': '1u', 'w6': '7.5u', 'l6': '1u',
                'w7': '150u', 'l7': '1u', 'w8': '15u', 'l8': '1u', 'c1': '1p', 'r1': '1k'
            }
        ],
        "metrics": [
            {
                'gain': 50.37,
                'cmrr': 55.15,
                'gbw': 95.52,
                'pm': 96.35
            },
            {
                'gain': 2.37,
                'cmrr': 25.15,
                'gbw': 35.52,
                'pm': 56.35
            }
        ],
        "targets": [
            9.2,
            9.3
        ],
        "aux_info": ["", ""]
    }

    # Test of the prompts generated by the proposer;
    examples = p.generate_examples(data_collected)
    #print(examples)
    # prompt_input = p.generate_desired_target(data_collected)
    prompt_input = ""
    prompt = p.generate_prompt(examples, prompt_input)

    #print(prompt)

    # Test of getting response from the GPT;

    params_proposed = p.propose_params(data_collected)

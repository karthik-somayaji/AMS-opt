import numpy as np
import re
import sys

sys.path.append("../")
from core.fewshot_agent import FewShotAgent # noqa: E402
from langchain.prompts import PromptTemplate, FewShotPromptTemplate


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

        related_ckts = args.related_ckts
        model = args.model

        # related_ckts = [ 'Two_Stage_Differential_Amplifier']
        # related_ckts = [  'Hysteresis_Comparator', 'Two_Stage_Differential_Amplifier']
        #related_ckts = [ 'Low_Dropout_Regulator']
        #related_ckts =  [ 'Hysteresis_Comparator',   'Low_Dropout_Regulator', 'Two_Stage_Differential_Amplifier']#, 'Low_Dropout_Regulator']#['Two_Stage_Differential_Amplifier', 'Hysteresis_Comparator']#['Two_Stage_Differential_Amplifier']##, 'Folded_Cascode_Amplifier', 'Hysteresis_Comparator']

        # Read from NATURAL LANGUAGE txt file
        multiline_history = """ """
        # models = [
        #     # "llama3-70B-instruct", 
        # "DeepSeek-R1-Distill-Llama-70B",
        # # "DeepSeek-R1-Distill-Qwen-32B",
        # # "Mistral-Small-24B-Instruct-2501",
        # # "Qwen2.5-32B-Instruct",
        # # "GPT-4o"
        # ]
        for ckt in related_ckts:
            ckt_KG_name = f'refined_{ckt}_{model}_KG.txt' if args.refined else f'1_{ckt}_{model}_KG.txt'
            #ckt_KG_name = f'combined_{ckt}_KG.txt' if args.refined else f'1_{ckt}_{model}_KG.txt'
        #with open(f'/home/karthik/sim_clean/LLMBO/history_summary/KG_{self.ckt_name_description}.txt', 'r') as file:
        #with open(f'/home/karthik/sim_clean/LLMBO/history_summary/{self.ckt_name_description}.txt', 'r') as file:
            # with open(f'/home/karthik/sim_clean/LLMBO/history_summary/critic_{ckt}.txt', 'r') as file:
            with open(f'/home/karthik/sim_clean/LLMBO/history_summary/{ckt_KG_name}', 'r') as file:
            #with open(f'/home/karthik/sim_clean/LLMBO/history_summary/KG_old_{ckt}.txt', 'r') as file:
            #with open(f'/home/karthik/sim_clean/LLMBO/history_summary/KG_bad_{ckt}.txt', 'r') as file:
            # Read the entire file content into a string
            #pass
                multiline_history += KG_prefix[ckt] + '\n' + file.read()

        # Read from CYPHER txt file
        with open('/home/karthik/sim_clean/LLMBO/history_summary/Two_Stage_Differential_Amplifier_cypher.txt', 'r') as file:
            # Read the entire file content into a string
            #multiline_history = file.read()
            pass

        # Comment if running first circuit or to independently optimize each circuit
        if (self.num_calls > 1):
            prefix += ""
            if args.history:
                # print(multiline_history)
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

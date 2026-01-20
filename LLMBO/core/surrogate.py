import numpy as np
from scipy.stats import norm
import re
import sys

sys.path.append("../")
from core.fewshot_agent import FewShotAgent # noqa: E402


class LazySurrogate(object):
    def __init__(self):

        pass

    def select_params_proposed(self, data_collected, params_proposed):

        return params_proposed[0]


class Surrogate(FewShotAgent):
    def __init__(self,
                 params_list,
                 task_context,
                 backend,
                 n_pred=5,
                 ranges=None,
                 use_aux_info=False,
                 max_request_attempt=3
                 ):
        super().__init__(
            use_aux_info=use_aux_info
        )
        self.task_context = task_context
        self.params_list = params_list
        self.params_units = ["f", "m", "n", "p", "u", "k", "G", "M"]
        self.n_params = len(self.params_list)

        self.backend = backend
        self.max_request_attempt = max_request_attempt # TODO: change request to adapt to larger number of proposal;

        self.ranges = ranges
        self.n_pred = n_pred
        self.lower_is_better = False

        self.debug_mode = False

    def select_params_proposed(self, data_collected, params_proposed):
        examples = self.sample(data_collected)

        prompts = []
        for params_proposed_i in params_proposed:
            prompt = self.generate_prompt(examples, params_proposed_i)
            prompts.append(prompt)

        # Predict the target value and calculates the variance for each proposed parameter settings;
        target_pred_mean = []
        target_pred_std = []
        for prompt in prompts:
            target_pred_i = []
            for _ in range(self.max_request_attempt):
                responses = self.backend.request(prompt)
                target_parsed = self.parse_llm_responses(responses)
                target_pred_i += target_parsed

                if self.debug_mode:
                    print("***Prompt***\n")
                    print(f"{prompt}")

                    print("***Response***\n")
                    print(f"{responses[0]}")

                if len(target_pred_i) >= self.n_pred:
                    break

            if len(target_pred_i) < self.n_pred:
                raise ValueError("LLM failed to make enough predictions!!")
            else:
                target_pred_i = np.array(target_pred_i[0:self.n_pred])

            # Calculate the mean and variance of the predicted target;
            target_pred_i_mean = np.mean(target_pred_i)
            target_pred_i_std = np.std(target_pred_i)

            target_pred_mean.append(target_pred_i_mean)
            target_pred_std.append(target_pred_i_std)

        # Using acquisition function to select the next params for query;
        params_query_idx = self.acquisition(target_pred_mean, target_pred_std, data_collected)

        return params_proposed[params_query_idx]

    def generate_prefix(self, prompt_input):
        prefix = self.task_context
        prefix += "**Examples** gives demonstration of existing parameter settings and the simulated target values from HSPICE. "
        # prefix += f"Predict the target value for the proposed parameters at your best attempt, even without direct access to the SPICE simulator. "
        # Additional requirements;
        # prefix += "Do not predict rounded values. Predict values with the highest possible precision. "
        # prefix += "Your response must only contain the predicted target value.\n"
        prefix += "**Examples**\n"

        return prefix

    def generate_suffix(self, prompt_input):
        # suffix = """params: {prompt_input}\n target value:"""
        suffix = "Based on the netlist and demonstrations, what is the prediction of the target value of the belew parameter setting:\n"
        suffix += "Your response must only contain the predicted target value in the format of target values in **Examples**.\n"
        suffix += "params: {prompt_input}"""
        return suffix

    def parse_llm_responses(self, responses):
        # TODO: range of the params is not checked currently;
        # TODO: add support to log the LLM reasoning part;

        # Extract formatted sub-strings with the format: param=value_param with regular expression;
        # value_param needs to be float number (optionally) followed by one of the units;
        def extract_params(response):
            match = re.search(r'target=([-+]?\d*\.\d+|\d+)', response)
            if match:
                # Store the found value using the parameter as the key;
                return float(match.group(1))
            else:
                # If any parameter is not found or its value is invalid, return False;
                return False

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

    # def parse_llm_responses(self, responses):
    #     target_parsed = []
    #     # Response is None means api call has failed in request attempts (default to be 3);
    #     if responses is None:
    #         print(f"Api call fails!")
    #     else:
    #         for response in responses:
    #             if self.validate_llm_response(response):
    #                 target_parsed.append(float(response))
    #
    #     return target_parsed
    #
    # def validate_llm_response(self, response):
    #     # TODO: need to add regular expression support for this function;
    #     try:
    #         # Convert the string representation to a list, then to a NumPy array
    #         data = float(response)
    #         return True
    #
    #     except Exception as e:
    #         if self.debug_mode:
    #             print(f"Validating{response} encounters error:{e}")
    #         return False

    def acquisition(self, target_pred_mean, target_pred_std, data_collected):
        target_collected = data_collected["targets"]

        target_pred_mean = np.array(target_pred_mean)
        target_pred_std = np.array(target_pred_std)

        if self.lower_is_better:
            target_best = np.min(target_collected)
            delta = -1*(target_pred_mean - target_best)
        else:
            target_best = np.max(target_collected)
            delta = target_pred_mean - target_best

        with np.errstate(divide='ignore'):  # handle y_std=0 without warning
            Z = delta / target_pred_std

        ei = np.where(target_pred_mean > 0, delta * norm.cdf(Z) + target_pred_mean * norm.pdf(Z), 0)

        params_query_idx = np.argmax(ei)
        return params_query_idx


if __name__ == "__main__":
    from backend.llm.gpt import GPT
    from core.proposer import Proposer

    task_context = """
    You are an analog circuit design expert and thinks step by step to solve the design problem. **Circuit Netlist** gives an HSPICE netlist of a two-stage differential amplifier. 

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
    p = Proposer(
        params_list=["w1", "l1", "w2", "l2", "w3", "l3", "w4", "l4", "w5", "l5", "w6", "l6", "w7", "l7", "w8", "l8",
                     "c1", "r1"],
        n_proposal=5,
        task_context=task_context,
        backend=b,
    )
    s = Surrogate(
        params_list=["w1", "l1", "w2", "l2", "w3", "l3", "w4", "l4", "w5", "l5", "w6", "l6", "w7", "l7", "w8", "l8",
                     "c1", "r1"],
        task_context=task_context,
        backend=b
    )
    data_collected = {
        "params": [
            ".param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=7u l5=1u w6=7u l6=1u w7=150u l7=1u w8=15u l8=1u c1=1p r1=1k",
            ".param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=7.5u l5=1u w6=7.5u l6=1u w7=150u l7=1u w8=15u l8=1u c1=1p r1=1k",
            ".param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=7.5u l5=1u w6=7.5u l6=1u w7=150u l7=1u w8=15u l8=1u c1=1p r1=1k",
            ".param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=6u l5=1u w6=6u l6=1u w7=150u l7=1u w8=15u l8=1u c1=1p r1=1k",
            ".param w1=200u l1=2u w2=200u l2=2u w3=20u l3=1u w4=20u l4=1u w5=10u l5=1u w6=10u l6=1u w7=200u l7=2u w8=20u l8=2u c1=1p r1=1k"
        ],
        "metrics": np.random.randn(5, 4),
        "targets": [
            9.2,
            9.35,
            9.1,
            10.3,
            9.8
        ],
        "aux_info": [[''] for _ in range(5)]
    }

    # Test of the prompts of the surrogate using proposed params from the proposer;
    params_proposed = [
        '.param w1=160u l1=1.5u w2=160u l2=1.5u w3=16u l3=1u w4=16u l4=1u w5=8u l5=1u w6=8u l6=1u w7=160u l7=1.5u w8=16u l8=1.5u c1=1p r1=1k',
        '.param w1=180u l1=1.5u w2=180u l2=1.5u w3=18u l3=1u w4=18u l4=1u w5=9u l5=1u w6=9u l6=1u w7=180u l7=1.5u w8=18u l8=1.5u c1=1p r1=1k',
        '.param w1=180u l1=1.5u w2=180u l2=1.5u w3=18u l3=1.2u w4=18u l4=1.2u w5=8u l5=1u w6=8u l6=1u w7=180u l7=1.5u w8=18u l8=1.5u c1=1p r1=1k',
        '.param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=6.5u l5=1u w6=6.5u l6=1u w7=150u l7=1u w8=15u l8=1u c1=1p r1=1k',
        '.param w1=180u l1=1.5u w2=180u l2=1.5u w3=18u l3=1u w4=18u l4=1u w5=9u l5=1u w6=9u l6=1u w7=180u l7=1.5u w8=18u l8=1.5u c1=1p r1=1k'
    ]


    # Test of getting response from the GPT;
    examples = s.sample(data_collected)

    prompts = []
    for params_proposed_i in params_proposed:
        prompt = s.generate_prompt(examples, params_proposed_i)
        prompts.append(prompt)

    # Test of selecting next params for query using the surrogate;
    params_query = s.select_params_proposed(data_collected, params_proposed)


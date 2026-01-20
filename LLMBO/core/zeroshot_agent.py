from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import re
import sys

sys.path.append('../')


class ZeroShotAgent(object):
    def __init__(self,
                 task_name,
                 params_list,
                 task_context,
                 backend,
                 max_request_attempt=3,
                 ):
        
        self.task_name = task_name
        self.task_context = task_context
        self.params_list = params_list
        self.params_units = ["f", "m", "n", "p", "u", "k", "G", "M"]
        self.n_params = len(self.params_list)

        self.backend = backend
        self.max_request_attempt = max_request_attempt

        self.debug_mode = False

    def generate_prompt(self, n_init_data):
        """generate a prompt for zero-shot warmstart.

        Args:
            n_init_data (int): the number of initial data points to be generated.
        """
        # prefix
        prompt = self.task_context
        prompt += "Based on the netlist and demonstrations, summarize how you can adjust the sizing of parameters to achievea high target values. "
        prompt += f"Conclude your response only with {n_init_data} recommended sets of parameters in the format of params in **Examples**. \n"
        # prompt += "You must not add any comments beyond the recommendation.\n"
        prompt += "**Examples**\n"

        # For two-stage differential amplifier:
        if ('amp' in self.task_name):
            print('IN  AMP')
            # example
            #prompt += ".param w1=[]\n .param l1=[]\n .param w2=[]\n .param l2=[]\n .param w3=[]\n .param l3=[]\n .param w4=[]\n .param l4=[]\n .param w5=[]\n .param l5=[]\n \
                #.param w6=[]\n .param l6=[]\n .param w7=[]\n .param l7=[]\n .param w8=[]\n .param l8=[]\n .param c1=[]\n .param r1=[]\n"
            prompt += ".param l1=[] w1=[] l2=[] w2=[] l3=[] w3=[] l4=[] w4=[] l5=[] w5=[] l6=[] w6=[] l7=[] w7=[] l8=[] w8=[] r1=[] c1=[] \n"

            # suffix
            prompt += "You need to replace [] with the actual values of the parameters, within a predefined normalized design space in the range [0,1].\n"
            prompt += f"Transistor width (w*): ranges from 0.12 micro meter to 50 micro meter ; where 0 corresponds to 0.12 micro meter and 1 corresponds to 50 micro meter \n"
            prompt += f"Transistor length (l*): ranges from 0.08 micro meter to 1 micro meter ; where 0 corresponds to 0.08 micro meter and 1 corresponds to 1 micro meter \n"
            prompt += f"Resistance (r*): ranges from 0.01 kilo Ohms to 100 Kilo Ohms ; where 0 corresponds to 0.01 kilo Ohms and 1 corresponds to 100 kilo Ohms \n"
            prompt += f"Capacitance (c*): ranges from 0.01 pico Farads to 100 pico Farads ;  where 0 corresponds to 0.01 pico Farads and 1 corresponds to 0.01 pico Farads\n"
            #prompt += f"Your {n_init_data} design choices should aim for high target values in the differential amplifier circuit.\n"
            prompt += f"Your {n_init_data} design choices should provide diverse configurations within the specified design space, allowing for exploration of different design options while aiming for high target values in the differential amplifier circuit.\n"

        elif ('fc' in self.task_name):
            print('IN  FC')
            # example
            prompt += ".param l1=[] w1=[] l2=[] w2=[] l3=[] w3=[] l4=[] w4=[] l5=[] w5=[] l6=[] w6=[] l7=[] w7=[] l8=[] w8=[] l9=[] w9=[] l10=[] w10=[] l11=[] w11=[] l12=[] w12=[] l13=[] w13=[] l14=[] w14=[] l15=[] w15=[] r1=[] r2=[] \n"
            # suffix
            prompt += "You need to replace [] with the actual values of the parameters, within a predefined design space:\n"
            prompt += f"Transistor width (w*): 0.09u to 148.5u \n"
            prompt += f"Transistor length (l*): 0.09u to 0.9u \n"
            prompt += f"Resistance (r*): 10k to 100k \n"
            prompt += f"Do not explain your answer. Just output the {n_init_data} design choices. Your {n_init_data} design choices should aim for high target values in the folded cascode amplifier circuit. Note that your suggestion of the design point needs to begin with .param\n"
            # prompt += f"Your {n_init_data} design choices should provide diverse configurations within the specified design space, allowing for exploration of different design options while aiming for high target values in the differential amplifier circuit.\n"

        elif ('comp' in self.task_name):
            print('IN  comp')
            # example
            prompt += ".param l1=[] w1=[] l2=[] w2=[] l3=[] w3=[] l4=[] w4=[] l5=[] w5=[] l6=[] w6=[] l7=[] w7=[] l8=[] w8=[] l9=[] w9=[] l10=[] w10=[] l11=[] w11=[] l12=[] w12=[]  \n"
            # suffix
            prompt += "You need to replace [] with the actual values of the parameters, within a predefined design space:\n"
            prompt += f"Transistor width (w*): 0.09u to 148.5u \n"
            prompt += f"Transistor length (l*): 0.09u to 0.9u \n"
            prompt += f"Do not explain your answer. Just output the {n_init_data} design choices. Your {n_init_data} design choices should aim for high target values in the hysteresis comparator circuit. Note that your suggestion of the design point needs to begin with .param and param values must be in the range [0.0, 1.0].\n"
            # prompt += f"Your {n_init_data} design choices should provide diverse configurations within the specified design space, allowing for exploration of different design options while aiming for high target values in the differential amplifier circuit.\n"

        elif ('ldo' in self.task_name):
            print('IN  LDO')
            # example
            prompt += ".param l1=[] w1=[] l2=[] w2=[] l3=[] w3=[] l4=[] w4=[] l5=[] w5=[] l6=[] w6=[] l7=[] w7=[] l8=[] w8=[] l9=[] w9=[] l10=[] w10=[] l11=[] w11=[] l12=[] w12=[] l13=[] w13=[] l14=[] w14=[] l15=[] w15=[] l16=[] w16=[] l17=[] w17=[] l18=[] w18=[] l19=[] w19=[] l20=[] w20=[] \n"
            # suffix
            prompt += "You need to replace [] with the actual values of the parameters, within a predefined design space:\n"
            prompt += f"Transistor width (w*): 0.09u to 148.5u \n"
            prompt += f"Transistor length (l*): 0.09u to 0.9u \n"
            prompt += f"Do not explain your answer. Just output the {n_init_data} design choices. Your {n_init_data} design choices should aim for high target values in the low dropout regular circuit. Note that your suggestion of the design point needs to begin with .param and param values must be in the range [0.0, 1.0]. Please output {n_init_data} .param suggestions, not just one.\n"
            # prompt += f"Your {n_init_data} design choices should provide diverse configurations within the specified design space, allowing for exploration of different design options while aiming for high target values in the differential amplifier circuit.\n"

        elif ('dcdc' in self.task_name):
            print('IN  DCDC')
            # example
            #prompt += ".param l1=[] w1=[] l2=[] w2=[] l3=[] w3=[] l4=[] w4=[] l5=[] w5=[] l6=[] w6=[] l7=[] w7=[] l8=[] w8=[] l9=[] w9=[] l10=[] w10=[] l11=[] w11=[] l12=[] w12=[] \n"
            prompt += ".param l1=[] w1=[] l2=[] w2=[] l3=[] w3=[] l4=[] w4=[] l5=[] w5=[] l6=[] w6=[] l7=[] w7=[] l8=[] w8=[] l9=[] w9=[] l10=[] w10=[] l11=[] w11=[] l12=[] w12=[] l13=[] w13=[] l14=[] w14=[] l15=[] w15=[] l16=[] w16=[] l17=[] w17=[] l18=[] w18=[] l19=[] w19=[] l20=[] w20=[] l21=[] w21=[] l22=[] w22=[] \n"
            # suffix
            prompt += "You need to replace [] with the actual values of the parameters, within a predefined design space:\n"
            #prompt += f"Transistor width (w*): 0.09u to 148.5u \n"
            #prompt += f"Transistor length (l*): 0.09u to 0.9u \n"
            prompt += f"Do not explain your answer. Just output the {n_init_data} design choices. Your {n_init_data} design choices should aim for high target values in the dcdc converter circuit. Note that your suggestion of the design point needs to begin with .param and param values must be in the range [0.0, 1.0]. Please output {n_init_data} .param suggestions, not just one.\n"
            # prompt += f"Your {n_init_data} design choices should provide diverse configurations within the specified design space, allowing for exploration of different design options while aiming for high target values in the differential amplifier circuit.\n"




        return prompt

    def extract_params(self, response):

        # Pattern to match parameter names followed by equals sign and floating point numbers
        pattern = rf"(\w+)\s*=\s*([\d\.-]+)"
        # Find all occurrences of the pattern
        matches = re.findall(pattern, response)
        # Convert matches to dictionary with parameter names as keys and float values as values
        param_dict = {param: value for param, value in matches}
        return param_dict


    def parse_llm_responses(self, responses):
        """A function to parse the LLM response for the warmstart.
            it split the response into lines by '\n'keyward and search for the .param keyword; then selects the corresponding sentences;
            in each selected sentence, it extracts the parameters and their values, and formats them into a string.

        Args:
            responses ([str]): a list of string responses from the LLM.
        Returns:
            params_parsed ([str]): a list of parsed parameters.
        """
        # Extract for every collected response;
        params_parsed = []
        # Response is None means api call has failed in request attempts (default to be 3);
        if responses is None:
            print(f"Api call fails!")
        else:
            responses = responses[0].split("\n")
            # search for the .param keyword and select the corresponding sentences;
            for response in responses:
                if ".param" in response:
                    response_parsed = self.extract_params(response)
                    if response_parsed:
                        params_parsed.append(response_parsed)

        return params_parsed

    def propose_params(self, n_init_data):

        params_init = []

        for _ in range(self.max_request_attempt):
            prompt = self.generate_prompt(n_init_data)

            responses = self.backend.request(prompt)
            print(n_init_data)
            print(responses)

            params_init = self.parse_llm_responses(responses)

            if self.debug_mode:
                print("***Prompt***\n")
                print(f"{prompt}")

                print("***Response***\n")
                print(f"{responses[0]}")

            #print(params_init)
            #print(len(params_init), n_init_data)

            if len(params_init) >= n_init_data:
                break

        if len(params_init) < n_init_data:
            raise ValueError("LLM failed to propose enough valid data!!")
        else:
            params_init = params_init[0:n_init_data]
            return params_init
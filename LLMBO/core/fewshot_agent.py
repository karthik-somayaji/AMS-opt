from langchain.prompts import PromptTemplate, FewShotPromptTemplate


class FewShotAgent(object):
    def __init__(self,
                 example_keys,
                 ):
        """
        params_list: list of size [n_params], containing names of all parameters;
        data_collected:
            params: list of size [n_data_collected, n_params]: string in format:
                    .param {name_param_1}={value_param_1} {name_param_2}={value_param_2} {name_param_n}={value_param_n}
            aux_info: list of size [n_data_collected]: auxiliary information in each query;

        value_param: value in real number or HSPICE supported format;
        """
        self.example_keys = example_keys

        example_template = ""
        for example_key in example_keys:
            example_template += f"{example_key}: {{{example_key}}}\n"

        self.example_prompt = PromptTemplate(
            input_variables=self.example_keys,
            template=example_template
        )

    def generate_prefix(self, prompt_input):
        raise NotImplementedError

    def generate_suffix(self, prompt_input):
        raise NotImplementedError

    def generate_prefix_for_history(self):
        raise NotImplementedError
    
    def generate_suffix_for_history(self):
        raise NotImplementedError
    
    def generate_prompt_for_history(self, examples, prompt_input):
        prefix = self.generate_prefix_for_history(prompt_input)
        suffix = self.generate_suffix_for_history(prompt_input)

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["prompt_input"],
            example_separator="\n"
        )

        prompt = few_shot_prompt.format(prompt_input=f"{prompt_input}")
        print(' *********************HISTORY PROMPT *****************: ')
        #print(prompt)
        return prompt

    def generate_prompt(self, examples, prompt_input, args):
        prefix = self.generate_prefix(prompt_input, args)
        suffix = self.generate_suffix(prompt_input)

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["prompt_input"],
            example_separator="\n"
        )
        #print(prompt_input, ' FEW SHOT PROMPT', few_shot_prompt)

        prompt = few_shot_prompt.format(prompt_input=f"{prompt_input}")
        #print(prompt)
        return prompt

    def generate_examples(self, data_collected):
        n_data_collected = len(data_collected["params"])
        examples = []

        for i in range(n_data_collected):
            example = {}
            for example_key in self.example_keys:
                example_str = ""
                # For params: assemble the named  parameters;
                if example_key == "params":
                    for k, v in data_collected["params"][i].items():
                        example_str += f"{k}={v} "
                elif example_key == "metrics":
                    for k, v in data_collected["metrics"][i].items():
                        example_str += f"{k}={v:.3f} "
                elif example_key == "targets":
                    example_str = f"target={data_collected['targets'][i]:.3f}"
                elif example_key == "aux_info":
                    #pass
                    example_str = data_collected["aux_info"]
                else:
                    raise ValueError("Invalid example key!")
                example[example_key] = example_str
            examples.append(example)

        return examples
    
    def generate_examples_for_history(self, data_collected):
        n_data_collected = len(data_collected["params"])
        examples = []

        for i in range(n_data_collected):
            example = {}
            for example_key in self.example_keys:
                example_str = ""
                # For params: assemble the named  parameters;
                if example_key == "params":
                    for k, v in data_collected["params"][i].items():
                        example_str += f"{k}={v} "
                elif example_key == "metrics":
                    for k, v in data_collected["metrics"][i].items():
                        example_str += f"{k}={v:.3f} "
                elif example_key == "targets":
                    example_str = f"target={data_collected['targets'][i]:.3f}"
                elif example_key == "aux_info":
                    #pass
                    example_str = data_collected["aux_info"]
                else:
                    raise ValueError("Invalid example key!")
                example[example_key] = example_str
            examples.append(example)

        return examples

    def parse_llm_responses(self, responses):
        raise NotImplementedError


if __name__ == "__main__":

    import re


    def extract_parameters(input_string, parameters, units):
        # Initialize a dictionary to store the found values
        found_values = {}

        # Create a regular expression pattern for the units, allowing only specified units
        units_pattern = f"[{''.join(re.escape(unit) for unit in units)}]?"  # Escape units to avoid regex issues

        # Iterate through each parameter and search for it in the input string
        for param in parameters:
            # Regex to find the parameter followed by a number (including decimal numbers) and optionally a unit, allowing spaces
            match = re.search(rf"{re.escape(param)}\s*=\s*(\d+(\.\d+)?{units_pattern})\b", input_string)
            if match:
                # Store the found value using the parameter as the key
                found_values[param] = match.group(1)
            else:
                # If any parameter is not found or its value is invalid, return False
                return False

        formatted_output = ".param " + " ".join(f"{param}={value}" for param, value in found_values.items())

        # Collect values in the order of the parameters list
        return formatted_output
    

    def parse_llm_responses(self, responses):
        raise NotImplementedError


if __name__ == "__main__":

    import re


    def extract_parameters(input_string, parameters, units):
        # Initialize a dictionary to store the found values
        found_values = {}

        # Create a regular expression pattern for the units, allowing only specified units
        units_pattern = f"[{''.join(re.escape(unit) for unit in units)}]?"  # Escape units to avoid regex issues

        # Iterate through each parameter and search for it in the input string
        for param in parameters:
            # Regex to find the parameter followed by a number (including decimal numbers) and optionally a unit, allowing spaces
            match = re.search(rf"{re.escape(param)}\s*=\s*(\d+(\.\d+)?{units_pattern})\b", input_string)
            if match:
                # Store the found value using the parameter as the key
                found_values[param] = match.group(1)
            else:
                # If any parameter is not found or its value is invalid, return False
                return False

        formatted_output = ".param " + " ".join(f"{param}={value}" for param, value in found_values.items())

        # Collect values in the order of the parameters list
        return formatted_output


    # Example usage
    units = ["f", "m", "n", "p", "u", "k", "G", "M"]
    parameters = ["w1", "l1", "w2", "l2", "w3", "l3", "w4", "l4", "w5", "l5", "w6", "l6", "w7", "l7", "w8", "l8", "c1", "r1"]
    input_string = "params: .param w1=200u, l1=1u, w2=200u, l2=1u, w3=20u, l3=1u, w4=20u, l4=1u, w5=10u, l5=1u, w6=10u, l6=1u, w7=200u, l7=1u, w8=20u, l8=1u, r1=1k, c1=1p"
    result = extract_parameters(input_string, parameters, units)
    print(result)


    #
    # import re
    #
    #
    # def extract_parameters(input_string, parameters, units):
    #     # Initialize a dictionary to store the found values
    #     found_values = {}
    #
    #     # Create a regular expression pattern for the units, allowing only specified units
    #     units_pattern = f"[{''.join(re.escape(unit) for unit in units)}]?"  # Escape units to avoid regex issues
    #
    #     # Iterate through each parameter and search for it in the input string
    #     for param in parameters:
    #         # Regex to find the parameter followed by a number (including decimal numbers) and optionally a unit, allowing spaces
    #         match = re.search(rf"{re.escape(param)}\s*=\s*(\d+(\.\d+)?{units_pattern})\b", input_string)
    #         if match:
    #             # Store the found value using the parameter as the key
    #             found_values[param] = match.group(1)
    #         else:
    #             # If any parameter is not found or its value is invalid, return False
    #             return False
    #
    #     # Collect values in the order of the parameters list
    #     return tuple(found_values[param] for param in parameters)
    #
    #
    # units = ["f", "m", "n", "p", "u", "k", "G", "M"]
    # parameters = ["w1", "l1", "w2", "l2", "c1", "r1"]
    #
    # input_string = "w1=10.2u, l1=20m, w2=30k, l2=40G, c1=50.96M, r1=0.22f"
    # result = extract_parameters(input_string, parameters, units)
    # print(result)
    #
    # input_string = "w1=10u, l1=20m, w2=30k, l2=40G, c1=50M"
    # result = extract_parameters(input_string, parameters, units)
    # print(result)
    #
    # input_string = "w1=10u, l1=20m, w2=30k, l2=40G, c1=50M, r1=60b"
    # result = extract_parameters(input_string, parameters, units)
    # print(result)
    #
    # input_string = "w1=10, l1=20, w2=30, l2=40, c1=50, r1=60"
    # result = extract_parameters(input_string, parameters, units)
    # print(result)
    #
    # input_string = "w1=10u, l1=20x, w2=30k, l2=40G, c1=50M, r1=60"
    # result = extract_parameters(input_string, parameters, units)
    # print(result)
    #
    # input_string = "w1=10um, l1=20mp, w2=30k, l2=40G, c1=50M, r1=60f"
    # result = extract_parameters(input_string, parameters, units)
    # print(result)
    #
    # input_string = "w1= 10u, l1 =20m , w2 = 30k, l2= 40G , c1=50M , r1=60f"
    # result = extract_parameters(input_string, parameters, units)
    # print(result)
    #
    # input_string = "r1=60f, c1=50M, l2=40G, w2=30k, l1=20m, w1=10u"
    # result = extract_parameters(input_string, parameters, units)
    # print(result)
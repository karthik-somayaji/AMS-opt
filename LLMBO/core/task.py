import numpy as np
import json
import utils
import sys

sys.path.append("../")
from backend.spice.interface import hspice_eval_f # noqa: E402
from backend.spice_FC.interface import hspice_eval_f_FC # noqa: E402
from backend.spice_comp.interface import hspice_eval_f_comp
from backend.spice_ldo.interface import hspice_eval_f_ldo
from backend.spice_dcdc.interface import hspice_eval_f_dcdc
from core.zeroshot_agent import ZeroShotAgent # noqa: E402
from backend.llm import gpt # noqa: E402


class Task(object):

    def __init__(self, path_task_setting, data_dict_keys):
        # Load json configuration files;
        with open(path_task_setting, 'r') as file:
            self.task_setting = json.load(file)
        self.data_dict_keys = data_dict_keys

        self.name_task = self.task_setting["ckt_name"]

        self.ckt_name_description = self.task_setting["ckt_name_description"]

        # Parameter list and metrics list;
        self.params_list = self.task_setting["params_list"]
        self.metrics_list = self.task_setting["metrics_list"]

        self.n_params = len(self.params_list)
        self.n_metrics = len(self.metrics_list)

        # Task context;
        self.task_context = ""

        with open(self.task_setting["path_description"], 'r') as file:
            task_description = file.read()

        self.task_context += task_description

        # Allowed range of parameters;
        self.ranges = {
            "w": self.task_setting["width_range"],
            "l": self.task_setting["length_range"],
            "r": self.task_setting["resistance_range"],
            "c": self.task_setting["capacitance_range"]
        }

        # Instantiate zeroshot warmup for further use;
        self.zeroshot_agent = ZeroShotAgent(
            task_name = self.name_task,
            params_list=self.params_list,
            task_context=self.task_context,
            backend=gpt.GPT(
                model="3.5", seed=114514, n_gen=1, max_token=1000
            ),
        )

    def generate_params_init(self, n_init_data, init_method):
        if init_method == "zeroshot":
            params_init = self.zeroshot_agent.propose_params(n_init_data)
        elif init_method == "random":
            params_init = self.random_initialize(n_init_data)
        elif init_method == "fixed":
            params_init = self.fixed_initialize(n_init_data) #TODO: need to change;
        else:
            raise ValueError("Initialization method not accepted!")

        return params_init

    def initialize(self, n_init_data, init_method, log_info=True):
        data_collected = utils.create_empty_data_dict(self.data_dict_keys)
        data_collected_llm = utils.create_empty_data_dict(self.data_dict_keys)
        data_collected_bo = utils.create_empty_data_dict(self.data_dict_keys)

        params_init = self.generate_params_init(n_init_data, init_method)

        for params_init_i in params_init:
            #print(params_init)
            data_init_i = self.evaluate(params_init_i, log_info=log_info)
            for k, v in data_init_i.items():
                data_collected[k].append(v)
        return data_collected, data_collected_llm, data_collected_bo

    def random_initialize(self, n_init_data):
        """generate random initial data from uniform distribution

        Args:
            n_init_data (int): the number of initial data to generate
        """
        params_init = []
        for i in range(n_init_data):
            params_numpy_i = np.zeros(self.n_params)

            for j, param in enumerate(self.params_list):
                if "w" in param[0]:
                    params_numpy_i[j] = np.random.uniform(self.task_setting["width_range"][0],
                                                          self.task_setting["width_range"][1])
                elif "l" in param[0]:
                    params_numpy_i[j] = np.random.uniform(self.task_setting["length_range"][0],
                                                          self.task_setting["length_range"][1])
                elif "r" in param[0]:
                    params_numpy_i[j] = np.random.uniform(self.task_setting["resistance_range"][0],
                                                          self.task_setting["resistance_range"][1])
                elif "c" in param[0]:
                    params_numpy_i[j] = np.random.uniform(self.task_setting["capacitance_range"][0],
                                                          self.task_setting["capacitance_range"][1])

            param_dict_i = utils.nparray_to_params_dict(params_numpy_i, self.params_list)
            params_init.append(param_dict_i)

        return params_init

    def fixed_initialize(self, n_init_data):
        """
        Generate the initial params, annotation, and f values.
        This can be loading existing data or query the language model as in LLAMBO.
        """
        data_init = {
            "params": [],
            "metrics": [],
            "targets": [],
            "aux_info": [],
            "params_numpy": []
        }

        params_init_pool = [
            {
                'w1': '25u', 'l1': '2.5u', 'w2': '20u', 'l2': '2.5u', 'w3': '25u', 'l3': '2u',
                'w4': '20u', 'l4': '2u', 'w5': '10u', 'l5': '2u', 'w6': '25u', 'l6': '2u',
                'w7': '30u', 'l7': '2.5u', 'w8': '2u', 'l8': '2u', 'r1': '20k', 'c1': '0.2p'
            }
            ]
        for i in range(n_init_data):
            data_new = self.evaluate(params_init_pool[i])
            for k, v in data_new.items():
                data_init[k].append(v)

        return data_init

    def evaluate(self, params_query, log_info=False):

        #print('params_query ', params_query)

        # Convert params_query (named dictionary into param.inc format);
        #params_inc = ".param " + "".join(f"{param}={value} " for param, value in params_query.items())
        params_inc = "".join(f".param {param}={value}\n" for param, value in params_query.items())

        # Simulation with HSPICE;
        if('amp' in self.name_task):
            _, f_eval_raw = hspice_eval_f(params_inc, self.task_setting)
        elif('fc' in self.name_task):
            _, f_eval_raw = hspice_eval_f_FC(params_inc, self.task_setting)
        elif('comp' in self.name_task):
            _, f_eval_raw = hspice_eval_f_comp(params_inc, self.task_setting)
        elif('ldo' in self.name_task):
            _, f_eval_raw = hspice_eval_f_ldo(params_inc, self.task_setting)
        elif('dcdc' in self.name_task):
            _, f_eval_raw = hspice_eval_f_dcdc(params_inc, self.task_setting)


        # Post simulation processing: gather results in a dictionary;
        params_numpy = utils.params_dict_to_nparray(params_query)
        metrics = self.gather_metrics(f_eval_raw)
        target = f_eval_raw['fom']
        aux_info = f_eval_raw['aux_info']

        data_new = {
            "params": params_query,
            "metrics": metrics,
            "targets": target,
            "aux_info": aux_info,
            "params_numpy": params_numpy
        }

        if log_info:
            metrics = " " + "".join(f"{k}={v:.3f} " for k, v in metrics.items())
            print(f"params: {params_inc}")
            print(f"metrics: {metrics}")
            print(f"targets: {target:.3f}")
            print(f"aux_info: {aux_info}")

        return data_new

    def gather_metrics(self, fvals):
        metrics = {}
        for metric in self.metrics_list:
            metrics[metric] = fvals[metric]

        return metrics

# main function
if __name__ == '__main__':
    path_task_setting = '/home/yuwang/Coding/LLMBO/tasks/amp2/amp2.json'
    task = Task(path_task_setting, data_dict_keys=["params", "metrics", "targets", "aux_info", "params_numpy"])
    params_query = {'w1': '120u', 'l1': '1.2u', 'w2': '120u', 'l2': '1.2u', 'w3': '20u', 'l3': '1.2u', 'w4': '20u', 'l4': '1.2u', 'w5': '10u', 'l5': '1.2u', 'w6': '10u', 'l6': '1.2u', 'w7': '120u', 'l7': '1.2u', 'w8': '20u', 'l8': '1.2u', 'c1': '1.2p', 'r1': '1.2k'}
    print(params_query)
    data_new = task.evaluate(params_query)

    _, _, _ = task.initialize(5, init_method="zeroshot")


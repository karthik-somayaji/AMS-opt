import torch
import os
import numpy as np
import pickle

from backend.llm import gpt
from core import task
from core import proposer
from core import bo
from core import sampler
from core import surrogate
import history_summary
import utils
import argparse


def _resolve_llmbo_relative_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    llmbo_root = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(llmbo_root, path)
    return candidate



class LLMBO(object):
    def __init__(self,
                 path_task_setting,
                 n_init_data=1,
                 init_method="zeroshot",
                 #n_sample=10,
                 n_sample=3,
                 sample_method="topk",
                 shuffle_sample=False,
                 n_proposal_llm=1,
                 n_proposal_bo=1,
                 n_itr=10,
                 gpt_version="3.5",
                 openai_api_seed=95,
                 # openai_api_seed=42,
                 path_checkpoints="./checkpoints",
                 rank_based_on_bo=True
                 ):
        """
        Args:
            n_init_data: number of datapoints for initialization;
            init_method: initialization method, should be ["zeroshot", "random", "fixed"]
            n_sample: number of high quality data to be sampled from data_collected;
            sample_method: sample method, should be ["topk", "random", "all"]
            shuffle_sample: whether to shuffle the sample in each iteration;
            n_proposal_llm: number of parameter settings proposed by LLM in each iteration, 0 means do not use llm proposer;
            n_proposal_bo: number of parameter settings proposed by BO (GP) in each iteration, 0 means do not use bo proposer;
        """
        assert init_method in ["zeroshot", "random", "fixed"]
        assert sample_method in ["topk", "random", "all", "mixed"]
        if gpt_version != "3.5" and gpt_version != "4":
            raise ValueError(f"gpt_version expect to be '3.5' or '4', got '{gpt_version}'.")

        self.gpt_version = gpt_version
        self.openai_api_seed = openai_api_seed
        if os.environ.get("OPENAI_API_KEY"):
            self.backend = gpt.GPT(model=gpt_version, seed=openai_api_seed, debug_mode=False)
        else:
            self.backend = None
            if init_method == "zeroshot":
                print("[LLMBO] OPENAI_API_KEY not set; falling back init_method='random'.")
                self.init_method = "random"

        self.data_dict_keys = ["params", "metrics", "targets", "aux_info", "params_numpy"]
        self.task = task.Task(path_task_setting, self.data_dict_keys)

        self.name_task = self.task.name_task
        self.ckt_name_description = self.task.ckt_name_description

        self.params_list = self.task.params_list
        self.n_params = self.task.n_params
        self.n_metrics = self.task.n_metrics
        self.ranges = self.task.ranges

        self.n_init_data = n_init_data
        if not hasattr(self, "init_method"):
            self.init_method = init_method
        self.n_sample = n_sample
        self.sample_method = sample_method
        self.shuffle_sample = shuffle_sample

        self.n_proposal_llm = min(5, n_proposal_llm) if rank_based_on_bo==True else 1 #n_proposal_llm
        self.n_proposal_bo = n_proposal_bo

        self.rank_based_on_bo = rank_based_on_bo

        self.n_itr = n_itr

        self.path_checkpoints = path_checkpoints + f"_{self.name_task}_gpt{gpt_version}"
        if not os.path.exists(self.path_checkpoints):
            os.makedirs(self.path_checkpoints)

        self.checkpoints = {
            "gpt_version": self.gpt_version,
            "openai_api_seed": openai_api_seed
        }

        self.data_collected = utils.create_empty_data_dict(self.data_dict_keys)
        self.data_collected_llm = utils.create_empty_data_dict(self.data_dict_keys)
        self.data_collected_bo = utils.create_empty_data_dict(self.data_dict_keys)

        self.params_proposed = []


        # Get initial data and add them to collected data;
        self.data_collected, self.data_collected_llm, self.data_collected_bo = self.task.initialize(self.n_init_data, self.init_method, log_info=True)


        self.llm_proposer = None
        if self.backend is not None and self.n_proposal_llm > 0:
            self.llm_proposer = proposer.LLMProposer(
                ckt_name_description=self.task.ckt_name_description,
                params_list=self.params_list,
                task_context=self.task.task_context,
                backend=self.backend,
                n_proposal=self.n_proposal_llm,
                ranges=self.ranges,
                example_keys=["params", "metrics", "targets", "aux_info"],
            )

        self.sampler = sampler.Sampler(
            n_sample=self.n_sample,
            method=self.sample_method,
            shuffle=self.shuffle_sample
        )

        self.bo_proposer = bo.GPBO(
            data_init=self.data_collected,
            params_list=self.params_list,
            n_proposal=self.n_proposal_bo,
            ranges=self.ranges
        )

        self.target_best = max(self.data_collected["targets"])
        print(f"Initial best target value: {self.target_best:.2f}")

    def optimize(self, args):
        print("Beginning Design Cycle")
        fom_array = []
        for itr in range(self.n_itr):
            print(f"Current iteration:{itr+1}.")
            # Sample high quality data for llm using sampler;
            data_pro = self.sampler.sample(self.data_collected)
            #print(data_pro['targets'])

            params_proposed_llm = []
            if self.llm_proposer is not None:
                params_proposed_llm = self.llm_proposer.propose_params(data_pro, args)

            print('************* IN HISTORY *************')            
            #responses_for_history = self.llm_proposer.generate_prompt_history_summary(data_pro) 
            print('************* OUT HISTORY *************')  

            params_proposed_bo = self.bo_proposer.propose_params(self.data_collected)

            if self.rank_based_on_bo and len(params_proposed_llm) > 0:
                #print(params_proposed_llm)
                params_proposed_llm_array = np.array([utils.params_dict_to_nparray(x) for x in params_proposed_llm ])
                ranked_llm_acq_fn_values = self.bo_proposer.obtain_acq_function_values(params_proposed_llm_array)
                top_k_indices = np.argsort(ranked_llm_acq_fn_values)[-1:]
                #top_k_values = params_proposed_llm_array[top_k_indices]
                params_proposed_llm = [utils.nparray_to_params_dict(params_proposed_llm_array[x], self.params_list) for x in top_k_indices]
                #params_proposed_llm = [utils.nparray_to_params_dict(params_proposed_llm_array[np.argmax(ranked_llm_acq_fn_values)], self.params_list)]

            for params_query_llm in params_proposed_llm:
                print('IN LLM LOOP')
                data_new_llm = self.task.evaluate(params_query_llm, log_info=False)
                self.update_data(self.data_collected_llm, data_new_llm)
                self.update_data(self.data_collected, data_new_llm)

            for params_query_bo in params_proposed_bo:
                print('IN BO LOOP')
                data_new_bo = self.task.evaluate(params_query_bo, log_info=False)
                self.update_data(self.data_collected_bo, data_new_bo)
                self.update_data(self.data_collected, data_new_bo)

            
            self.target_best = max(self.data_collected["targets"])
            self.target_best_index = self.data_collected["targets"].index(self.target_best)
            self.target_best_llm = max(self.data_collected_llm["targets"])
            self.target_best_index_llm = self.data_collected_llm["targets"].index(self.target_best_llm)
            self.target_best_bo = max(self.data_collected_bo["targets"])
            self.target_best_index_bo = self.data_collected_bo["targets"].index(self.target_best_bo)

            #print("Best opt. point",  self.data_collected["params"][self.target_best_index])
            #print(f"At iteration: {itr + 1}, the best target value is: {self.target_best:.3f}, bo best target: {self.target_best_bo:.3f}")
            #print("Best LLM opt. point" ,self.data_collected_llm["params"][self.target_best_index_llm])
            print("Best metrics",  self.data_collected["metrics"][self.target_best_index])
            print("Best BO metrics",  self.data_collected_bo["metrics"][self.target_best_index_bo])
            print("Best LLM metrics",  self.data_collected_llm["metrics"][self.target_best_index_llm])
            print(f"At iteration: {itr + 1}, the best target value is: {self.target_best:.3f}, llm best target: {self.target_best_llm:.3f}, bo best target: {self.target_best_bo:.3f}")

            fom_array.append(self.target_best)
            

            # Log current iteration information and save checkpoints; TODO:
            self.save_checkpoints(itr)

        with open(f"results/{args.history}_{args.refined}_comp_{args.model}.txt" , 'w') as f:
            print("Best metrics",  self.data_collected["metrics"][self.target_best_index], file=f)
            print("Best BO metrics",  self.data_collected_bo["metrics"][self.target_best_index_bo], file=f)
            print("Best LLM metrics",  self.data_collected_llm["metrics"][self.target_best_index_llm], file=f)
            print(f"At iteration: {itr + 1}, the best target value is: {self.target_best:.3f}, llm best target: {self.target_best_llm:.3f}, bo best target: {self.target_best_bo:.3f}", file=f)

        np.savetxt(f"results/foms_{args.history}_{args.refined}_comp_{args.model}.txt", np.array(fom_array), fmt="%.4f")


    def critic(self, file_name):
        with open(f'history_summary/{file_name}.txt', 'r') as file:
            content = file.read()

        prefix = """Consider you are an analog designer critic. Given a circuit type, netlist and description of design rules for a given circuit, your job is to judge the correctness of the design rules from the knowledge you possess about circuit design. Below is given one such circuit.

        *** Circuit Description with design rules ***
        
        """
        suffix = """Given the netlist and the design rules, please modify the design rules mention accordingly if you find factual inconsistencies and generate a response similar to the input, but with changed design rules if you think there are factual discrepancies in the design rules in the input. Please output only the corrected design rules for Optimizing parameters considering tradeoffs:."""

        prompt = prefix
        prompt += content
        prompt += suffix

        response = self.backend.request(prompt)

        with open(f'critic_{file_name}.txt', 'w') as file:
            file.write(response[0])

    @staticmethod
    def update_data(data_collected, data_new):
        for k, v in data_new.items():
            data_collected[k].append(v)


    def save_checkpoints(self, itr):
        self.checkpoints["data_collected"] = self.data_collected
        self.checkpoints["itr"] = itr + 1

        with open(self.path_checkpoints + '/checkpoints.pkl', 'wb') as f:
            pickle.dump(self.checkpoints, f)

    def load_checkpoints(self, itr):
        with open(self.path_checkpoints + '/checkpoints.pkl', 'rb') as f:
            checkpoints = pickle.load(f)

        return checkpoints


if __name__ == "__main__":

    # np.random.seed(14)
    # torch.random.manual_seed(14)
    np.random.seed(114)
    torch.random.manual_seed(114)
    openai_api_seed = 514

    parser = argparse.ArgumentParser(description="Simple argparse example")
    # String argument
    parser.add_argument('--model', type=str, help='Your name: DeepSeek-R1-Distill-Llama-70B')
    # Boolean flag (store_true means it becomes True if specified)
    parser.add_argument('--history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--refined', type=int, choices=[0, 1], default=0)
    # List argument (space-separated values)
    parser.add_argument('--related_ckts', nargs='+', help='List of items')
    args = parser.parse_args()

    # LLMBO + GPBO;
    llmbo = LLMBO(
        _resolve_llmbo_relative_path("tasks/amp2/amp2.json"),
        #"tasks/FC/FC.json",
        #"tasks/comp/comp.json",
        #"tasks/ldo/ldo.json",
        #"tasks/dcdc/dcdc.json",
        openai_api_seed=openai_api_seed,
        gpt_version="3.5",
        #n_init_data=5, # for amp
        n_init_data=3,
        #n_init_data=1,
        n_proposal_llm=4, # for amp
        #n_proposal_llm=3,
        n_proposal_bo=1,
        n_itr=30,
        # n_itr=20,
        rank_based_on_bo = False#True #True#True# True
    )


    # Only run with GPBO;
    # llmbo = LLMBO(
    #     "tasks/amp2/amp2.json",
    #     openai_api_seed=openai_api_seed,
    #     gpt_version="3.5",
    #     n_init_data=5,
    #     n_proposal_llm=0,
    #     n_proposal_bo=2,
    #     n_itr=10
    # )

    llmbo.optimize(args)

    # llmbo_critic = LLMBO(
    #     #"tasks/amp2/amp2.json",
    #     #"tasks/FC/FC.json",
    #     "tasks/comp/comp.json",
    #     #"tasks/ldo/ldo.json",
    #     openai_api_seed=openai_api_seed,
    #     gpt_version="4",
    #     n_init_data=4,
    #     #n_init_data=1,
    #     n_proposal_llm=1,
    #     n_proposal_bo=1,
    #     n_itr=30
    #     #n_itr=20
    # )

    # llmbo_critic.critic(file_name= 'Low_Dropout_Regulator')

    # python llmbo.py --history 1 --refined 1 --model DeepSeek-R1-Distill-Llama-70B --related_ckts ["Two_Stage_Differential_Amplifier"]
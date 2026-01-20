from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import os, sys
sys.path.append("../")
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
from core.task import Task
from utils.parser import numpy2params
from utils.dataset import update_data_collected, get_best_example
# mute warnings
import warnings
warnings.filterwarnings("ignore")


class GPBO(object):
    def __init__(self, path_task_setting:str, n_init_data:int=10, n_query:int=10, n_candidates:int=1, data_collected=None):
        """ Gaussian Process Bayesian Optimization

        Args:
            path_task_setting (str): the path of the task setting json file
            n_init_data (int, optional): the number of initial data. Defaults to 10.
            n_query (int, optional): the number of queries. Defaults to 10.
            n_candidates (int, optional): the number of candidates. Defaults to 1.
            data_collected (dict, optional): the collected data dictionary. Defaults to None.
        """
        self.task = Task(path_task_setting)
        self.n_init_data = n_init_data
        self.n_query = n_query
        self.n_candidates = n_candidates

        self.name_task = self.task.name_task
        self.n_params = self.task.n_params
        self.n_metrics = self.task.n_metrics
        # self.use_absolute_size = self.task.use_absolute_size
        self.ranges = self.task.ranges



        # Get initial data and add them to collected data;
        if data_collected is None:
            self.data_collected = self.task.random_initialization(self.n_init_data)
        else:
            self.data_collected = data_collected

        self.target_best = np.max(self.data_collected["targets"][:self.n_init_data])
        print(f"Initial best target value: {self.target_best:.2f}")

        # initialize gp model
        self.initialize_model()

    def get_x_bounds(self):
        """
        get the bounds of the parameters


        Returns:
            torch.Tensor: 2 x d tensor, where the first row is the minimum value of each parameter and the second row is the maximum value of each parameter
        """
        bounds = torch.zeros(2, self.n_params)
        for j, param in enumerate(self.task.params_list):
            if "w" in param[0]:
                bounds[0, j] = self.task.ranges["w"][0]
                bounds[1, j] = self.task.ranges["w"][1]
            elif "l" in param[0]:
                bounds[0, j] = self.task.ranges["l"][0]
                bounds[1, j] = self.task.ranges["l"][1]
            elif "r" in param[0]:
                bounds[0, j] = self.task.ranges["r"][0]
                bounds[1, j] = self.task.ranges["r"][1]
            elif "c" in param[0]:
                bounds[0, j] = self.task.ranges["c"][0]
                bounds[1, j] = self.task.ranges["c"][1]
        return bounds

    def initialize_model(self, state_dict=None):
        data_collected = self.data_collected
        train_x = torch.tensor(data_collected["params_numpy"], dtype=torch.double)
        train_obj = torch.tensor(data_collected["targets"], dtype=torch.double).reshape(-1, 1)
        # standardize data
        train_obj = standardize(train_obj)
        # define models for objective and constraint
        gp = SingleTaskGP(train_x, train_obj, covar_module=ScaleKernel(RBFKernel(ard_num_dims=self.n_params)))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        # load state dict if it is passed
        if state_dict is not None:
            gp.load_state_dict(state_dict)
        self.gp = gp
        self.mll = mll
    def propose(self):
        # collect data
        data_collected = self.data_collected
        train_obj = torch.tensor(data_collected["targets"], dtype=torch.double)
        # initialize model
        # standardize data
        train_obj = standardize(train_obj)
        self.initialize_model(self.gp.state_dict())
        # fit model
        fit_gpytorch_mll(self.mll)
        # define acquisition function
        if self.n_params == 1:
            EI = ExpectedImprovement(self.gp, train_obj.max())
        else:
            EI = qExpectedImprovement(self.gp, train_obj.max())
        # optimize acquisition function
        candidates, acq_value = optimize_acqf(EI, bounds=self.get_x_bounds(), q=self.n_candidates, num_restarts=5, raw_samples=20, sequential=True)
        return candidates.numpy()

    def evaluate(self, params_query):
        data_new = self.task.evaluate(params_query)
        return data_new

    def optimize(self):
        for itr in range(0, self.n_query, self.n_candidates):
            print(f"Current iteration:{itr+1}.")

            candidates = self.propose()
            for i, candidate in enumerate(candidates):
                params_query = numpy2params(candidate, self.task.params_list)
                print(f"HSPICE simulation with params: {params_query}")

                data_new = self.task.evaluate(params_query)

                # Update the collected data;
                self.data_collected = update_data_collected(self.data_collected, data_new)



                target_new = data_new["targets"]
                self.target_best = np.max(self.data_collected["targets"])
                print(f"At iteration {itr+1+i}, the queried data is {target_new:.6f}, the best target value is: {self.target_best:.6f}")
        # print the best target value and the corresponding data, as scientific notation, with 2 decimal places
        return self.data_collected

if __name__ == "__main__":

    np.random.seed(114)
    torch.random.manual_seed(114)
    openai_api_seed = 514
    np.set_printoptions(precision=2, suppress=True)
    bo = GPBO("/tasks/amp2/amp2.json", n_init_data=3, n_query=15, n_candidates=5)
    data_collected = bo.optimize()
    best_example = get_best_example(data_collected)
    print(f"The best example is:{best_example}")
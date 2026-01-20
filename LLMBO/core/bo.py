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
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, qExpectedImprovement, AcquisitionFunction
from botorch.optim import optimize_acqf
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np

from utils.parser import nparray_to_params_dict

# mute warnings
import warnings
warnings.filterwarnings("ignore")

class LowerConfidenceBound(AcquisitionFunction):
    def __init__(self, model, kappa=5.0):
        super().__init__(model=model)
        self.kappa = kappa
        self.model = model

    def forward(self, X):
        X = X.squeeze(1)
        posterior = self.model.posterior(X) 
        #samples = posterior.rsample()
        mean = posterior.mean  # Ensure mean is [n] for n point
        sigma = posterior.variance.sqrt()
        lcb = mean + self.kappa * sigma
        return lcb

class GPBO(object):
    def __init__(self,
                 data_init,
                 params_list,
                 n_proposal=10,
                 ranges=None
                 ):
        """ Gaussian Process Bayesian Optimization

        Args:
            path_task_setting (str): the path of the task setting json file
            n_init_data (int, optional): the number of initial data. Defaults to 10.
            n_query (int, optional): the number of queries. Defaults to 10.
            n_proposal (int, optional): the number of candidates. Defaults to 1.
            data_collected (dict, optional): the collected data dictionary. Defaults to None.
        """
        self.params_list = params_list
        self.n_proposal = n_proposal
        self.n_params = len(self.params_list)
        self.ranges = ranges

        # initialize gp model
        print("Initializing BO GP with data_init...")
        self.initialize_model(data_init)

    def get_x_bounds(self):
        """
        get the bounds of the parameters

        Returns:
            torch.Tensor: 2 x d tensor, where the first row is the minimum value of each parameter and the second row is the maximum value of each parameter
        """
        bounds = torch.zeros(2, self.n_params)
        for j, param in enumerate(self.params_list):
            if "w" in param[0]:
                bounds[0, j] = self.ranges["w"][0]
                bounds[1, j] = self.ranges["w"][1]
            elif "l" in param[0]:
                bounds[0, j] = self.ranges["l"][0]
                bounds[1, j] = self.ranges["l"][1]
            elif "r" in param[0]:
                bounds[0, j] = self.ranges["r"][0]
                bounds[1, j] = self.ranges["r"][1]
            elif "c" in param[0]:
                bounds[0, j] = self.ranges["c"][0]
                bounds[1, j] = self.ranges["c"][1]
        return bounds

    def initialize_model(self, data_collected, state_dict=None):
        params_numpy_list = data_collected["params_numpy"]

        # print(params_numpy_list)
        # print((params_numpy_list[0]))

        train_x = torch.tensor(params_numpy_list)#torch.tensor(np.array(params_numpy_list), dtype=torch.double)
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

    def obtain_acq_function_values(self, new_point):

        # print(new_point)
        # print(new_point.shape)

        self.gp, train_obj_max = self.acquisition_function[0], self.acquisition_function[1]
        #EI = qExpectedImprovement(self.gp, train_obj_max)# beta = 1.0)#, train_obj_max)
        lcb = LowerConfidenceBound(self.gp, kappa=0.25)
        
        acq_vals = lcb(torch.tensor(new_point, dtype=torch.float32)).squeeze()#.unsqueeze(1))

        # print(acq_vals)

        return acq_vals.detach().numpy()



    def propose_params(self, data_collected):
        if self.n_proposal == 0:
            return []
        else:
            train_obj = torch.tensor(data_collected["targets"], dtype=torch.double)
            # initialize model
            # standardize data
            train_obj = standardize(train_obj)
            self.initialize_model(data_collected=data_collected, state_dict=self.gp.state_dict())
            # fit model
            fit_gpytorch_mll(self.mll)
            # define acquisition function
            if self.n_params == 1:
                EI = ExpectedImprovement(self.gp, train_obj.max())
            else:
                EI = qExpectedImprovement(self.gp, train_obj.max())

            # save current acquisition function
            self.acquisition_function = [self.gp, train_obj.max()]

            # optimize acquisition function
            candidates, acq_value = optimize_acqf(EI, bounds=self.get_x_bounds(), q=self.n_proposal, num_restarts=5, raw_samples=20, sequential=True)

            params_proposed_numpy = candidates.numpy()

            # Convert to dictionary;
            params_proposed = []
            for i, candidate in enumerate(params_proposed_numpy):
                params_query = nparray_to_params_dict(candidate, self.params_list)
                params_proposed.append(params_query)

            return params_proposed


if __name__ == "__main__":

    np.random.seed(114)
    torch.random.manual_seed(114)
    openai_api_seed = 514
    np.set_printoptions(precision=2, suppress=True)


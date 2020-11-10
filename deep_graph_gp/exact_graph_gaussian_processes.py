import torch
import numpy as np
import dgl
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import lazify, delazify

class ExactGraphGP(ExactGP):
    def __init__(self, likelihood, mean, kernel):
        super(ExactGraphGP, self).__init__(None, None, likelihood)
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, g, x=None):
        """

        """
        if x is None:
            x = g.ndata["h"]

        mean = self.mean_module(x)
        cov  = self.covar_module(x)
        cov.add_jitter()

        adj = g.adjacency_matrix(False) # Adjacency matrix
        d   = g.out_degrees().float() # Degree vector
        n   = adj.shape[0]    # Number of vertices

        a_plus_i = torch.eye(n) + adj
        m = torch.matmul(a_plus_i, mean.view(mean.numel(),1))
        m /= (d.view(d.numel(),1) + 1.)

        S = delazify(cov)

        S = torch.matmul(a_plus_i, torch.matmul(S, a_plus_i.T))
        # print(S)
        # print(torch.cholesky(S))
        S = S * (torch.ger(1./(d+1.), 1./(d+1.)))
        return MultivariateNormal(m, S)
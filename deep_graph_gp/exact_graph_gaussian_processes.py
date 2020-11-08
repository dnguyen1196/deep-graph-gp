import torch
import numpy as np
import dgl
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal


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
        adj = dgl.khop_adj(g, 1) # Adjacency matrix
        d   = g.out_degrees() # Degree vector
        n   = adj.shape[0]

        a_plus_i = adj + torch.eye(n)
        
        m = torch.dot(a_plus_i, mean) / (d[:, None]+1)

        S = (a_plus_i @ cov) @ a_plus_i
        S = S * torch.outer(d +1, d+1)

        return MultivariateNormal(m, S)
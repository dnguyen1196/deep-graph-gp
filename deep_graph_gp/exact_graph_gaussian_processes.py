import torch
import numpy as np
import dgl
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import lazify, delazify
from torch_sparse.tensor import SparseTensor


class ExactGraphGP(ExactGP):
    def __init__(self, likelihood, mean, kernel):
        super(ExactGraphGP, self).__init__(None, None, likelihood)
        self.mean_module = mean
        self.covar_module = kernel


    def sparse_adj_matmul(self, adj, v):
        """[summary]

        :param adj: [description]
        :type adj: [type]
        :param v: [description]
        :type v: [type]
        :return: [description]
        :rtype: [type]
        """
        return adj.matmul(v)

    def forward(self, x, g, **kwargs):
        """

        """
        # Pre-compute (I+D)^{-1} (A+I) (multiply each row of A+I with (1+di)^-1)
        adj_mat_filled_diag = SparseTensor.from_torch_sparse_coo_tensor(g.adjacency_matrix(False)).fill_diag(1.)
        adj_mat_filled_diag = adj_mat_filled_diag / adj_mat_filled_diag.sum(-1).unsqueeze(-1) # Divide each row by (1+di)

        if torch.cuda.is_available() and x.is_cuda:
            adj_mat_filled_diag = adj_mat_filled_diag.cuda()

        # This will be of shape [num_output_dim, nx, nx] -> Prohibitive for big nx
        covar_xx = self.covar_module(x).evaluate()

        # covar_xx = (I+D)^{-1} (A+I) K_xx (A+I)^top (I+D)^{-1}
        # First compute  (I+D)^{-1} (A+I) @ K_xx
        xx_t1 = self.sparse_adj_matmul(adj_mat_filled_diag, covar_xx)
        # Then compute  (I+D)^{-1} (A+I) @ ((A+I) @ K_xx).T = (A+I) @ K_xx @ (A+I).T
        covar_full = self.sparse_adj_matmul(adj_mat_filled_diag, xx_t1.transpose(-2, -1))

        mean_full = self.mean_module(x)
        mean_full = self.sparse_adj_matmul(adj_mat_filled_diag, mean_full)
        
        return gpytorch.distributions.MultivariateNormal(mean_full, covar_full)
import torch
import numpy as np
import dgl
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, IndependentMultitaskVariationalStrategy
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.lazy import lazify
from torch_sparse.tensor import SparseTensor
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch import settings
from gpytorch.lazy import BlockDiagLazyTensor
from gpytorch.models import ApproximateGP

"""
How do we avoid using the entire adjacency matrix in our calculation?

Can we instead of passing in the entire adjacency matrix of the original
graph, pass in the subgraph (during batch) training?

"""

class VariationalGraphGP(ApproximateGP):
    def __init__(self, inducing_points, in_dim, out_dim=None, mean=None, covar=None, is_output_layer=True):
        """[summary]

        :param inducing_points: [description]
        :type inducing_points: [type]
        :param mean: [description]
        :type mean: [type]
        :param covar: [description]
        :type covar: [type]
        :param num_output_dim: [description]
        :type num_output_dim: [type]
        :param full_x: [description], defaults to None
        :type full_x: [type], optional
        :param sparse_adj_mat: [description], defaults to None
        :type sparse_adj_mat: [type], optional
        """
    
        if out_dim is None:
            batch_shape = torch.Size([])
        else:
            batch_shape = torch.Size([out_dim])
        
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), 
            batch_shape=batch_shape
        )

        # LMCVariationalStrategy for introducing correlation among tasks
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=out_dim,
        )

        super(VariationalGraphGP, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape) if mean is None else mean
        self.covar_module = gpytorch.kernels.PolynomialKernel(power=3, batch_shape=batch_shape) if covar is None else covar
        self.num_inducing = inducing_points.size(-2)
        self.is_output_layer = is_output_layer

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

    def forward(self, x, g=None, x_inds=None):
        """[summary]

        :param x: [description]
        :type x: [type]
        :param g: [description]
        :type g: [dlg.DGLGraph]
        :param x_inds: [description], defaults to None
        :type x_inds: [type], optional
        :return: [description]
        :rtype: [type]
        """
        
        inducing_points = x[..., :self.num_inducing, :]
        inputs = x[..., self.num_inducing:, :]

        covar_zz = self.covar_module(inducing_points).evaluate()

        # all_x = x[..., self.num_inducing:, :].repeat(self.num_output_dim, 1, 1)
        # Pre-compute (I+D)^{-1} (A+I) (multiply each row of A+I with (1+di)^-1)
        adj_mat_filled_diag = SparseTensor.from_torch_sparse_coo_tensor(g.adjacency_matrix(False)).fill_diag(1.)
        adj_mat_filled_diag = adj_mat_filled_diag / adj_mat_filled_diag.sum(-1).unsqueeze(-1) # Divide each row by (1+di)

        # This will be of shape [num_output_dim, nx, nx] -> Prohibitive for big nx
        covar_xx_full = self.covar_module(inputs).evaluate()

        # This will be of shape [num_output_dim, nx, nz] -> Prohibitive for big nx
        covar_xz_full = self.covar_module(inputs, inducing_points).evaluate()

        # covar_xx = (I+D)^{-1} (A+I) K_xx (A+I)^top (I+D)^{-1}
        # First compute  (I+D)^{-1} (A+I) @ K_xx
        xx_t1 = self.sparse_adj_matmul(adj_mat_filled_diag, covar_xx_full)
        # Then compute  (I+D)^{-1} (A+I) @ ((A+I) @ K_xx).T = (A+I) @ K_xx @ (A+I).T
        xx_t2 = self.sparse_adj_matmul(adj_mat_filled_diag, xx_t1.transpose(-2, -1))

        # covar_xz = (I+D)^{-1} (A+I) K_xz
        xz_t1 = self.sparse_adj_matmul(adj_mat_filled_diag, covar_xz_full)
        
        # xx_t2 is shape [num_output_dim, nx, nx]
        # The following extracts the corresponding sub-tensor of shape
        # xx_t2[..., x_inds, :] will have shape [num_output_dim, len(x_inds), nx]
        # Then covar_xx will have shape
        # [num_output_dim, nx_batch, nx_batch]
        # If x_inds is None then 
        if x_inds is not None:
            covar_xx = xx_t2[..., x_inds, :][..., :, x_inds]
            covar_xz = xz_t1[..., x_inds, :]
        else:
            covar_xx = xx_t2[..., :, :]
            covar_xz = xz_t1[..., :, :]

        # Construct the covariance matrix
        covar_full = gpytorch.lazify(torch.cat([
                torch.cat([covar_zz, covar_xz.transpose(-2, -1)], dim=-1), 
                torch.cat([covar_xz, covar_xx], dim=-1)
            ], dim=-2)).add_jitter(1e-3)
        
        if x_inds is None:
            mean_full = self.mean_module(x)
        else:
            mean_full = self.mean_module(torch.cat([
                inducing_points, inputs[..., x_inds, :]
            ], dim=-2))

        return gpytorch.distributions.MultivariateNormal(mean_full, covar_full)
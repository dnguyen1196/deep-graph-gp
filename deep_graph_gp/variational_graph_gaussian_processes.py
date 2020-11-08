import torch
import numpy as np
import dgl
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import _VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor,\
    SumLazyTensor, delazify

"""

"""

class GraphVariationalStrategy(_VariationalStrategy):
    def __init__(self, model, inducing_points, variational_distribution):
        super(GraphVariationalStrategy).__init__(model, 
            inducing_points, variational_distribution, True)
    
    def forward(self, g, x, inducing_points, inducing_values, variational_inducing_covar):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`
        :param dgl.DGLGraph g: A graph or sampled subgraph
        :param torch.Tensor x: Locations :math:`\mathbf X` to get the
            variational posterior of the function values at.
        :param torch.Tensor inducing_points: Locations :math:`\mathbf Z` of the inducing points
        :param torch.Tensor inducing_values: Samples of the inducing function values :math:`\mathbf u`
            (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        :param ~gpytorch.lazy.LazyTensor variational_inducing_covar: If the distribuiton :math:`q(\mathbf u)`
            is Gaussian, then this variable is the covariance matrix of that Gaussian. Otherwise, it will be
            :attr:`None`.
        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        
        full_output = MultivariateNormal(
            self.model.mean_module(full_inputs),
            self.model.covar_module(full_inputs)
        )

        full_covar = full_output.lazy_covariance_matrix
        full_mean = full_output.mean
        
        adj = dgl.khop_adj(g, 1) # Adjacency matrix
        nnz = inducing_points.size(-2) # Number of inducing points
        nnx = adj.shape[0] # Number of test point
        A_tilde_upper_half = torch.cat([torch.zeros(nnz, nnz), torch.zeros(nnz, nnx)], dim=-1)
        A_tilde_lower_half = torch.cat([torch.zeros(nnx, nnz), adj], dim=-1)
        A = torch.cat([A_tilde_upper_half, A_tilde_lower_half], dim=-2)
        A_plus_i = A + np.eye(nnz + nnx)

        d   = torch.cat([torch.zeros(nnz), g.out_degrees()]) # Degree vector
        full_mean = torch.dot(A_plus_i, full_mean) / (d[:, None]+1)

        S = (A_plus_i @ full_covar) @ A_plus_i
        full_covar = S * torch.outer(d+1, d+1)

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_mean[..., num_induc:]

        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            del self._memoize_cache["cholesky_factor"]
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = torch.triangular_solve(
            induc_data_covar.double(), L, upper=False)[0].to(full_inputs.dtype)

        # Compute the mean of q(f)
        # Why K_ZZ^{-1/2}
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (
            torch.matmul(
                interp_term.transpose(-1, -2),
                (inducing_values - self.prior_distribution.mean).unsqueeze(-1)
            ).squeeze(-1)
            + test_mean
        )

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX 
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        middle_term = SumLazyTensor(variational_inducing_covar, middle_term)

        predictive_covar = SumLazyTensor(
            data_data_covar.add_jitter(1e-4),
            MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term),
        )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
    
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`p( \mathbf u)`
        """
        mean = self.model.mean_module(self.inducing_points)
        cov  = self.model.covar_module(self.inducing_points)
        return MultivariateNormal(mean, cov)


class VariationalGraphGP(ApproximateGP):
    def __init__(self, inducing_points, mean, covar):
        self.variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = GraphVariationalStrategy(self, inducing_points, 
            self.variational_distribution)
        
        super(VariationalGraphGP, self).__init__(variational_strategy)

        # The mean of the graph GP will be different as well!
        self.mean_module = mean
        self.covar_module = covar

    def forward(self, g, x=None):
        """
        :param 
        :param 
        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        q_u = self.variational_distribution.forward()
        if x is None:
            x = g.ndata["h"]
            
        return self.variational_strategy(g, x, self.inducing_points,
            q_u.mean, q_u.lazy_covariance_matrix)
    
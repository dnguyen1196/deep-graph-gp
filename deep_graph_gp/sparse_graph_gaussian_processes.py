import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, LMCVariationalStrategy, IndependentMultitaskVariationalStrategy

"""
By Jake Gardner, this implementation allows for multi-output 
It can be used either as part of multi-layer graph GP or 
for multiclass output together with a suitable likelihood such as

`gpytorch.likelihoods.SoftmaxLikelihood`

"""

class SparseGraphGP(ApproximateGP):
    def __init__(self, inducing_points, sparse_adj, full_x, num_latents, num_output_dim):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_latents]))
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True),
            num_tasks=num_output_dim,
        )
        super(SparseGraphGP, self).__init__(variational_strategy)
        self.num_latents = num_latents
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.PolynomialKernel(power=3, batch_shape=torch.Size([num_latents]))

        self.covar_module.offset = 5.
        
        self.sparse_adj = sparse_adj.fill_diag(1.)
        self.sparse_adj = self.sparse_adj / self.sparse_adj.sum(-1).unsqueeze(-1)
        print(self.sparse_adj.to_dense().shape)
        self.full_x = full_x
        self.num_inducing = inducing_points.size(-2)
        
    def sparse_adj_matmul(self, v):
        return self.sparse_adj.matmul(v)
        
    def forward(self, x, x_inds=None):
        print("model.forward")
        print(x_inds)
        inducing_points = x[:, :self.num_inducing, :]
        # inducing_points = self.inducing_points
        
        # print(inducing_points.shape)
        # print(x.shape)
        print("inducing points.shape", inducing_points.shape)
        print("x.shape", x.shape)
        covar_zz = self.covar_module(inducing_points).evaluate()
        covar_xx_full = self.covar_module(self.full_x.repeat(self.num_latents, 1, 1)).evaluate()
        covar_xz_full = self.covar_module(self.full_x.repeat(self.num_latents, 1, 1), inducing_points).evaluate()

        print("covar_zz.shape", covar_zz.shape)
        print("covar_xx_full.shape", covar_xx_full.shape)
        print("covar_xz_full.shape", covar_xz_full.shape)
        
        xx_t1 = self.sparse_adj_matmul(covar_xx_full)
        xx_t2 = self.sparse_adj_matmul(xx_t1.transpose(-2, -1))
        xz_t1 = self.sparse_adj_matmul(covar_xz_full)
        
        covar_xx = xx_t2[..., x_inds, :][..., :, x_inds]
        covar_xz = xz_t1[..., x_inds, :]
        
        print("covar_xx.shape", covar_xx.shape)
        print("covar_xz.shape", covar_xz.shape)

        covar_x = gpytorch.lazify(torch.cat([
                                torch.cat([covar_zz, covar_xz.transpose(-2, -1)], dim=-1), 
                                torch.cat([covar_xz, covar_xx], dim=-1)
                            ], dim=-2)).add_jitter(1e-3)
        
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
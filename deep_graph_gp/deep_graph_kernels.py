import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch.conv import GraphConv
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, IndependentMultitaskVariationalStrategy


class DeepGraphKernel(ApproximateGP):
    def __init__(self, input_dim, num_inducing, hidden_sizes=[32, 32], out_dim=None, mean=None, covar=None):

        if out_dim is None:
            batch_shape = torch.Size([])
        else:
            batch_shape = torch.Size([out_dim])
        
        if out_dim is None:
            inducing_points = torch.rand(num_inducing, hidden_sizes[-1])
        else:
            inducing_points = torch.rand(out_dim, num_inducing, hidden_sizes[-1])

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), 
            batch_shape=batch_shape
        )

        # Use LMCVariationalStrategy for introducing correlation among tasks
        if out_dim is None:
            variational_strategy = VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
        else:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=out_dim,
            )

        super(DeepGraphKernel, self).__init__(variational_strategy)

        gcn_layers = nn.ModuleList()
        layer_input_out_dims = list(zip(
            [input_dim] + hidden_sizes[:-1],
            hidden_sizes
        ))

        for i, (in_features, out_features) in enumerate(layer_input_out_dims):
            gcn_layers.append(
                GraphConv(in_features, out_features, activation=nn.ReLU())
            )

        self.mean_module = gpytorch.means.LinearMean(hidden_sizes[-1], batch_shape=torch.Size([out_dim])) if mean is None else mean
        self.covar_module = gpytorch.kernels.PolynomialKernel(power=4, batch_shape=batch_shape) if covar is None else covar
        # self.covar_module.offset = 5
        self.num_inducing = inducing_points.size(-2)
        self.gcn = gcn_layers
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, **kwargs):
        inducing_points = x[..., :self.num_inducing, :]
        inputs = x[..., self.num_inducing:, :]

        covar_hz = self.covar_module(inputs, inducing_points).evaluate()
        covar_zz = self.covar_module(inducing_points).evaluate()
        covar_hh = self.covar_module(inputs).evaluate()

        mean_full = self.mean_module(x)

        # Construct the covariance matrix
        covar_full = gpytorch.lazify(torch.cat([
                torch.cat([covar_zz, covar_hz.transpose(-2, -1)], dim=-1), 
                torch.cat([covar_hz, covar_hh], dim=-1)
            ], dim=-2)).add_jitter(1e-3)

        return gpytorch.distributions.MultivariateNormal(mean_full, covar_full)

    def graph_convolution(self, inputs, g):
        if isinstance(g, list):
            assert(len(g) == len(self.gcn))
            h = inputs
            for i, layer in enumerate(self.gcn):
                h = layer(g[i], h)
                if i != len(self.gcn)-1:
                    h = self.dropout(h)
            
        else:
            h = inputs
            for i, layer in enumerate(self.gcn):
                h = layer(g, h)
                if i != len(self.gcn)-1:
                    h = self.dropout(h)

        return h

    def __call__(self, inputs, g, prior=False, **kwargs):
        inputs = self.graph_convolution(inputs, g)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        return self.variational_strategy(inputs, prior=prior, **kwargs)
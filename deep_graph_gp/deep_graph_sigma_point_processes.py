import torch
import tqdm
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, BatchDecoupledVariationalStrategy
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from torch_sparse.tensor import SparseTensor
from gpytorch.lazy import BlockDiagLazyTensor
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP

class GraphDSPPLayer(DSPPLayer):
    def __init__(self, inducing_points, in_dim, out_dim=None, Q=8, mean=None, covar=None):
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
        
        # variational_distribution = CholeskyVariationalDistribution(
        #     inducing_points.size(-2), 
        #     batch_shape=batch_shape
        # )
        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
            batch_shape=torch.Size([out_dim]) if out_dim is not None else torch.Size([])
        )

        # Seems like it might be better to use independent multitask for single layer GP
        # And variational strategy for deep GP. They don't seem to play well together for some reason
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(GraphDSPPLayer, self).__init__(
            variational_strategy, in_dim, out_dim, Q)

        # TODO: make this modifiable
        self.mean_module =  gpytorch.means.LinearMean(in_dim, batch_shape=torch.Size([out_dim]))if mean is None else mean
        self.covar_module = gpytorch.kernels.PolynomialKernel(power=4, batch_shape=batch_shape) if covar is None else covar
        self.num_inducing = inducing_points.size(-2)

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

        if torch.cuda.is_available() and inputs.is_cuda:
            adj_mat_filled_diag = adj_mat_filled_diag.cuda()

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
        
        mean_z = self.mean_module(inducing_points)
        mean_x = self.mean_module(inputs)

        mean_x = mean_x.unsqueeze(-1)
        mean_x = self.sparse_adj_matmul(adj_mat_filled_diag, mean_x)

        mean_x = mean_x.squeeze(-1)

        if x_inds is not None:
            mean_x = mean_x[..., x_inds]
        
        mean_full = torch.cat([
                mean_z, mean_x
            ], dim=-1)

        return gpytorch.distributions.MultivariateNormal(mean_full, covar_full)


class DeepGraphSigmaPointProcesses(DSPP):
    def __init__(self, input_dim, first_layer_inducing_points=None,
                output_dim=None, num_inducing=128, Q=8,
                num_likelihood_samples=10,
                kernel_name="PolynomialKernel",
                kernel_params={},
                layer_dims=[10, 10]):
        
        super().__init__(Q)

        self.layers = torch.nn.ModuleList()
        layer_input_out_dims = list(zip(
            [input_dim] + layer_dims,
            layer_dims + [output_dim]
        ))

        self.num_likelihood_samples = num_likelihood_samples
        num_layers = len(layer_input_out_dims)
        for layer_ind, (in_dim, out_dim) in enumerate(layer_input_out_dims):            
            if layer_ind == 0 and first_layer_inducing_points is not None:
                assert(first_layer_inducing_points.shape[-1] == in_dim)
                assert(first_layer_inducing_points.shape[0] == out_dim)
                inducing_points = first_layer_inducing_points
            elif out_dim is None: # If output is just a scalar
                inducing_points = torch.randn(num_inducing, in_dim)
            else:
                inducing_points = torch.randn(out_dim, num_inducing, in_dim)
            
            # Add layer
            kernel_params["batch_shape"] = torch.Size([out_dim])
            if kernel_name == "PolynomialKernel":
                kernel_params["power"] = 3
            covar_func = getattr(gpytorch.kernels, kernel_name)(**kernel_params)

            self.layers.append(
                GraphDSPPLayer(inducing_points, in_dim=in_dim, out_dim=out_dim, Q=Q, covar=covar_func)
            )

    def forward(self, x, g, x_inds=None):
        """
        
        Supports stochastic training for large graphs
        
        See: https://docs.dgl.ai/guide/minibatch-node.html#guide-minibatch-node-classification-sampler 
        for stochastic training with mini-batch subsampling in large graphs

        This sampler samples for "multiple-layer" of neighbor hoods
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)

        dataloader = dgl.dataloading.NodeDataLoader(
            g, train_inds, sampler,
            batch_size=32,
            shuffle=True,
            drop_last=False,
            num_workers=1)

        input_nodes, output_nodes, blocks = next(iter(dataloader))
        where
        blocks is a list of graphs of decreasing size (narrowing neighborhood)

        :param x: [description]
        :type x: [type]
        :param g: [description]
        :type g: [type]
        :param x_inds: [description], defaults to None
        :type x_inds: [type], optional
        :return: [description]
        :rtype: [type]
        """
        def find(tensor, values):
            """Helper function: given a tensor and specified values
            Find for each value, the index of the first element == value

            :param tensor: [description]
            :type tensor: [type]
            :param values: [description]
            :type values: [type]
            :return: [description]
            :rtype: [type]
            """
            return torch.nonzero(tensor[..., None] == values)[:, 1]

        if isinstance(g, list): # If g is given as blocks from RandomGraphSampling
            assert(len(g) == len(self.layers)) # The number of layers must match
            h = x
            for i, layer in enumerate(self.layers):
                num_output = g[i]._node_frames[1]["_ID"].shape[0]
                h = layer(h, g=g[i], # The pick x_inds as the index of the nodes to output
                    x_inds=find(g[i]._node_frames[0]["_ID"], g[i]._node_frames[1]["_ID"]))

        else:
            h = x
            for i, layer in enumerate(self.layers):
                with gpytorch.settings.num_likelihood_samples(self.num_likelihood_samples):
                    h = layer(h, g=g, x_inds=x_inds) if i == len(self.layers)-1 else layer(h, g=g)

        return h


    def predict(self, x, blocks, likelihood, x_inds=None):
        # TODO: modify this to match graph input
        with settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            preds = likelihood(self(x, g=blocks, x_inds=x_inds, mean_input=x))
            mu = preds.mean.cpu()
            var = preds.variance.cpu()

            # Compute test log probability. The output of a DSPP is a weighted mixture of Q Gaussians,
            # with the Q weights specified by self.quad_weight_grid. The below code computes the log probability of each
            # test point under this mixture.

            # Step 1: Get log marginal for each Gaussian in the output mixture.
            base_batch_ll = likelihood.log_marginal(y_batch, self(x))

            # Step 2: Weight each log marginal by its quadrature weight in log space.
            deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll

            # Step 3: Take logsumexp over the mixture dimension, getting test log prob for each datapoint in the batch.
            batch_log_prob = deep_batch_ll.logsumexp(dim=0)
            ll = batch_log_prob.cpu()

            return mu, var, ll

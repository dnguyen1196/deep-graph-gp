import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch.conv import GraphConv


class DeepGraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[32, 32], *args, **kwargs):
        super(DeepGraphNeuralNetwork, self).__init__(*args, **kwargs)

        self.layers = nn.ModuleList()

        layer_input_out_dims = list(zip(
            [input_dim] + hidden_sizes,
            hidden_sizes + [output_dim]
        ))

        for i, (in_features, out_features) in enumerate(layer_input_out_dims):
            if i != len(layer_input_out_dims) - 1:
                activation = nn.ReLU()
            else:
                # The last layer will have softmax activation
                activation = nn.Softmax()

            self.layers.append(
                GraphConv(in_features, out_features, activation=activation)
            )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, g, x, **kwargs):
        if isinstance(g, list):
            assert(len(g) == len(self.layers))
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(g[i], h)
                if i != len(self.layers)-1:
                    h = self.dropout(h)
            
        else:
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
                if i != len(self.layers)-1:
                    h = self.dropout(h)

        return h
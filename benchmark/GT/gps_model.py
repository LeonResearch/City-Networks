import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)

from .layer.gps_layer import GPSLayer



class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, dim_in, dim_out, dim_inner, layers, local_gnn_type, n_heads, dropout):
        super().__init__()

        self.pre_mp = torch.nn.Linear(dim_in, dim_inner)
        self.layers = torch.nn.ModuleList()
        for _ in range(layers):
            self.layers.append(GPSLayer(
                dim_h=dim_inner,
                local_gnn_type=local_gnn_type,
                num_heads=n_heads,
                dropout = dropout,
            ))

        self.post_mp = torch.nn.Linear(dim_inner, dim_out)
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.pre_mp.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, edge_index):
        if self.pre_mp is not None:
            x = self.pre_mp(x)
        for layer in self.layers:
            x, edge_index = layer(x, edge_index)
        
        x = self.post_mp(x)
        
        return x

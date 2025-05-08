import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
class ChebNet(nn.Module):
    def __init__(
            self, 
            in_channels,
            hidden_channels,
            out_channels,
            num_layers=2,
            dropout=0.5,
            K=2,   # K is the Chebyshev filter size
        ):
        super(ChebNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.K = K
        # Input projection (linear layer)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        # Chebyshev Convolutional layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K=self.K))
        # Output projection (linear layer)
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.SiLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        # Input projection
        x = self.input_proj(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Chebyshev Convolutional layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Output projection
        x = self.output_proj(x)
        return x
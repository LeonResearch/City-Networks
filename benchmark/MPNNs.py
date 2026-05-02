import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, ChebConv, GCN2Conv, PNAConv
import torch.nn as nn


class MPNNs(torch.nn.Module):
    def __init__(
            self, 
            gnn='gcn',
            in_channels=None,
            hidden_channels=None, 
            out_channels=None, 
            local_layers=3, 
            dropout=0.5, 
            heads=1, 
            pre_ln=False, 
            pre_linear=False, 
            res=False, 
            ln=False, 
            bn=False, 
            jk=False, 
            deg=None,
        ):
        super(MPNNs, self).__init__()
        self.gnn = gnn
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.pre_linear = True if gnn in ['gcnii'] else pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)

        self.aggregators = ['mean', 'min', 'max', 'std']
        self.scalers = ['identity', 'linear']

        if not self.pre_linear:
            if gnn=='gat':
                self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            elif gnn=='cheb':
                self.local_convs.append(ChebConv(in_channels, hidden_channels, K=2))
            elif gnn=='gcnii':
                ValueError("GCNII requires at least one pre-Linear layer")
            elif gnn=='mlp':
                self.local_convs.append(torch.nn.Linear(in_channels, hidden_channels))
            elif gnn=='pna':
                self.local_convs.append(
                    PNAConv(
                        in_channels, hidden_channels, 
                        aggregators=self.aggregators, scalers=self.scalers,
                        deg=deg, towers=1,
                    )
                )
            else:
                self.local_convs.append(GCNConv(in_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1

        for _ in range(local_layers):
            if gnn=='gat':
                self.local_convs.append(GATConv(hidden_channels, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            elif gnn=='sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif gnn=='cheb':
                self.local_convs.append(ChebConv(hidden_channels, hidden_channels, K=2))
            elif gnn=='gcnii':
                self.local_convs.append(GCN2Conv(hidden_channels, alpha=0.1, theta=0.5, layer=_ + 1))
            elif gnn=='mlp':
                self.local_convs.append(torch.nn.Linear(hidden_channels, hidden_channels))
            elif gnn=='pna':
                self.local_convs.append(
                    PNAConv(
                        hidden_channels, hidden_channels, 
                        aggregators=self.aggregators, scalers=self.scalers,
                        deg=deg, towers=1,
                    )
                )

            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels,
                        cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))
                
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()


    def forward(self, x, edge_index):
        x_final = 0

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x0 = x.clone()
        
        for i, local_conv in enumerate(self.local_convs):
            if self.gnn == 'gcnii':
                x = local_conv(x, x0, edge_index)
            elif self.gnn == 'mlp':
                if self.res:
                    x = local_conv(x) + self.lins[i](x)            
                else:
                    x = local_conv(x)
            else:
                if self.res:
                    x = local_conv(x, edge_index) + self.lins[i](x)
                else:
                    x = local_conv(x, edge_index)

            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x

        x = self.pred_local(x_final)

        return x

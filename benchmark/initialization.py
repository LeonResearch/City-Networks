# This file is modified based on the SGFormer repo
# link: https://github.com/qitianwu/SGFormer

from benchmark.MPNNs import MPNNs
from benchmark.GT.multi_model import MultiModel
from benchmark.GT.gps_model import GPSModel
from benchmark.GT.sgformer import SGFormer, GCN
from benchmark.GT.nodeformer import NodeFormer
from benchmark.GT.nagphormer import TransformerModel

def parse_method(configs, c, d, deg=None, device='cpu'):
    if configs['method'] in ['mlp', 'gcn', 'gat', 'sage', 'cheb', 'pna', 'gcnii']:
        model = MPNNs(
            gnn = configs['method'],
            in_channels=d,
            hidden_channels=configs['model']['hidden_channels'],
            out_channels=c,
            local_layers=configs['model']['num_layers'],
            dropout=configs['train']['dropout'],
            heads=configs['model']['num_heads'],
            pre_ln=configs['model']['pre_layer_norm'], 
            pre_linear=configs['model']['pre_linear'], 
            res=configs['model']['residual'], 
            ln=configs['model']['layer_norm'], 
            bn=configs['model']['batch_norm'], 
            jk=configs['model']['jump_knowledge'],
            deg=deg,
        ).to(device)
    elif configs['method'] == "gps":
        model = GPSModel(
            dim_in=d,
            dim_out=c,
            dim_inner=configs['model']['hidden_channels'],
            layers=configs['model']['num_layers'],
            local_gnn_type=configs['model']['gnn_type'], 
            n_heads=configs['model']['num_heads'], 
            dropout=configs['train']['dropout'],
        ).to(device)
    elif configs['method'] == "sgformer":
        gnn = GCN(
            in_channels=d,
            hidden_channels=configs['model']['hidden_channels'],
            out_channels=configs['model']['hidden_channels'],
            num_layers=configs['model']['num_layers'],
            dropout=configs['train']['dropout'],
        )
        model = SGFormer(
            gnn=gnn, 
            in_channels=d, 
            out_channels=c, 
            hidden_channels=configs['model']['hidden_channels'], 
            num_layers=configs['model']['num_layers'],
            alpha=configs['model']['alpha'], 
            dropout=configs['train']['dropout'],
            num_heads=configs['model']['num_heads'], 
            use_bn=configs['model']['use_bn'],
            use_residual=configs['model']['use_residual'], 
            use_graph=configs['model']['use_graph'],
            use_weight=configs['model']['use_weight'], 
            use_act=configs['model']['use_act'], 
            graph_weight=configs['model']['graph_weight'], 
            aggregate=configs['model']['aggregate'], 
            jk=configs['model']['jump_knowledge'],
        ).to(device)
    elif configs['method'] == 'exphormer':
        model = MultiModel(
            dim_in=d,
            dim_out=c,
            dim_inner=configs['model']['hidden_channels'],
            layers=configs['model']['num_layers'],
            local_gnn_type=configs['model']['gnn_type'], 
            n_heads=configs['model']['num_heads'], 
            dropout=configs['train']['dropout'],
        ).to(device)
    elif configs['method'] =='nodeformer':
        model = NodeFormer(
            in_channels=d,
            out_channels=c,
            hidden_channels=configs['model']['hidden_channels'],
            num_layers=configs['model']['num_layers'],
            dropout=configs['train']['dropout'],
            num_heads=configs['model']['num_heads'],
            use_bn=configs['model']['use_bn'],
        ).to(device)
    elif configs['method'] =='nagphormer':
        model = TransformerModel(
            hops=configs['model']['num_layers'], 
            n_class=c, 
            input_dim=d, 
            pe_dim = configs['model']['hidden_channels'],
            n_layers=configs['model']['num_layers'],
            num_heads=configs['model']['num_heads'],
            hidden_dim=configs['model']['hidden_channels'],
            ffn_dim=configs['model']['hidden_channels'],
            dropout_rate=configs['train']['dropout'],
            attention_dropout_rate=configs['train']['dropout'],
        ).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parser_add_main_args(parser):
    parser.add_argument('--exp_name', type=str, default='testing')
    parser.add_argument('--dataset', type=str, default='paris')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--runs', type=int, default=5, help='number of runs')
    parser.add_argument('--method', type=str, default='gcn', help='The model to use.')
    parser.add_argument('--num_layers', type=int, default=16, help='number of GNN layers')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--influence_dir', type=str, default='influence_results/testing')
    parser.add_argument('--num_samples_influence', type=int, default=200, 
                        help='number of samples to calculate the influence scores')
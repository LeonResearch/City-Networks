# This file is modified based on the SGFormer repo
# link: https://github.com/qitianwu/SGFormer

from benchmark.gnn import ChebNet
from benchmark.sgformer import SGFormer
from torch_geometric.nn import (
    GraphSAGE,
    MLP,
    GCN,
    GAT,
    MLP,
)

def parse_method(args, c, d, device):
    if args.method == 'mlp':
        model = MLP(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.method == 'gcn':
        model = GCN(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.method == 'gat':
        model = GAT(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.method == 'sage':
        model = GraphSAGE(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.method == 'cheb':
        model = ChebNet(
            in_channels=d,
            hidden_channels=args.hidden_channels,
            out_channels=c,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.method == 'sgformer':
        model = SGFormer(
            in_channels = d, 
            hidden_channels = args.hidden_channels, 
            out_channels = c,
            trans_num_layers=1, # always 1
            trans_num_heads=1, # always 1
            trans_dropout=args.transformer_dropout, 
            trans_use_bn=True, 
            trans_use_residual=True, 
            trans_use_weight=True, 
            trans_use_act=False, # always false
            gnn_num_layers=args.num_layers, 
            gnn_dropout=args.dropout, 
            gnn_use_weight=True, 
            gnn_use_init=False, # depends on the configuration, in our case set to False
            gnn_use_bn=True, 
            gnn_use_residual=True, 
            gnn_use_act=True,
            use_graph=True, 
            graph_weight=0.8, 
            aggregate='add',
        ).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='paris')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./citynetworks_data')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.1,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    parser.add_argument('--use_agumented_node_attr', action='store_true', help='use augmented node attr')
    parser.add_argument('--save_split', action='store_true', help='if to save the split mask')


    # model
    parser.add_argument('--method', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=16,
                        help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use batchnorm for each GNN layer')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')

    # sgformer
    parser.add_argument('--graph_weight', type=float,
                        default=0.8, help='graph weight.')
    parser.add_argument('--transformer_weight_decay', type=float, default=1e-5,
                        help='Ours\' weight decay. Default to weight_decay.')
    parser.add_argument('--ours_use_weight', action='store_true', help='use weight for trans convolution')
    parser.add_argument('--ours_use_bn', action='store_true', help='use layernorm for trans')
    parser.add_argument('--ours_use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--ours_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--ours_layers', type=int, default=2, help='gnn layer.')
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help='transformer dropout.')
    parser.add_argument('--aggregate', type=str, default='add', help='aggregate type, add or cat.')

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=10000, help='mini batch training for large graphs')
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience.')
    
    # display and utility
    parser.add_argument('--display_step', type=int, default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int, default=1, help='how often to evaluate')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--model_dir', type=str, default='../../model/')
    parser.add_argument('--experiment_name', type=str, default='testing')
    parser.add_argument('--num_sampling_worker', type=int, default=10, help='number of workers for NeighborLoader')
    parser.add_argument('--neighbors', type=list, nargs='+', default=[-1, -1, -1, -1, -1, -1, -1, -1],
                    help='List of neighbor values')
    
    # Influence score calculation 
    parser.add_argument('--influence_dir', type=str, default='influence_results/testing')
    parser.add_argument('--num_samples_influence', type=int, default=200, 
                        help='number of samples to calculate influence scores')
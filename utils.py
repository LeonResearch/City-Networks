import torch
from sklearn.metrics import f1_score
from torch_geometric.datasets import (
    Planetoid, 
    WikipediaNetwork, 
    HeterophilousGraphDataset,
    WebKB,
)
from ogb.nodeproppred import PygNodePropPredDataset, NodePropPredDataset
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.transforms import GDC
from torch_geometric.data import Data

from citynetworks import CityNetwork


def load_dataset(configs):
    print(f"Loading {configs['dataset']} Dataset ...")
    if configs['dataset'] in ["paris", "shanghai", "la", "london"]:
        dataset = CityNetwork(root=configs['dataset_dir'], name=configs['dataset'])
        data = dataset[0]

    elif configs['dataset'] in ["cora", "citeseer"]:
        dataset = Planetoid(root=configs['dataset_dir'], name=configs['dataset'])
        data = dataset[0]

        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
        data.edge_index = edge_index        
        data.x.fill_(1)

    elif configs['dataset'] in ["ogbn-arxiv"]:
        dataset = PygNodePropPredDataset(root=configs['dataset_dir'], name='ogbn-arxiv')
        data = dataset[0]
        # Some process similar to 
        # https://github.com/LUOyk1999/tunedGNN/blob/main/large_graph/dataset.py
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
        data.edge_index = edge_index
        splits = dataset.get_idx_split()
        data.train_mask = splits['train']
        data.val_mask = splits['valid']
        data.test_mask = splits['test']
        data.y = data.y.squeeze()
    elif configs['dataset'] in ["squirrel"]:
        dataset = WikipediaNetwork(
            root=configs['dataset_dir'],
            name=configs['dataset'],
        )
        data = dataset[0]
        split_idx = configs['seed']
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask = data.val_mask[:, split_idx]
        data.test_mask = data.test_mask[:, split_idx]
    elif configs['dataset'] in ['amazon-ratings', 'roman-empire']:
        dataset = HeterophilousGraphDataset(
            root=configs['dataset_dir'],
            name=configs['dataset'],
        )
        data = dataset[0]
        split_idx = configs['seed']
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask = data.val_mask[:, split_idx]
        data.test_mask = data.test_mask[:, split_idx]
    elif configs['dataset'] in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(
            root=configs['dataset_dir'],
            name=configs['dataset'],
        )
        data = dataset[0]
        split_idx = configs['seed']
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask = data.val_mask[:, split_idx]
        data.test_mask = data.test_mask[:, split_idx]

    else:
        raise ValueError("Current dataset is not implemented.")
    print(f"{configs['dataset']} Dataset Loaded!")
    return data


def eval_acc(output, labels):
    # Ensure preds is a 1D array of predicted label indices
    preds = output.argmax(dim=-1).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    # Compute macro F1 score and accuracy
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = (preds == labels).sum() / preds.shape[0] * 100
    return round(acc, 1), round(macro_f1, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Transfer edge attributes to node attributes
def attr_augmentation(data):
    node_edge_attr_sum = torch.zeros(data.x.shape[0], data.edge_attr.shape[1])                          
    node_edge_count = torch.zeros(data.x.shape[0], 1)
    # sum the corresponding edge attr for each node
    node_edge_attr_sum.index_add_(0, data.edge_index[1], data.edge_attr)
    node_edge_count.index_add_(
        0,
        data.edge_index[1],
        torch.ones_like(data.edge_index[1], dtype=torch.float).view(-1, 1)
    )
    # Prevent division by zero for nodes with no neighbors if the graph is not connected
    node_edge_count[node_edge_count == 0] = 1.0
    # Compute the mean of the neighbouring edge attributes for each node
    node_edge_attr_mean = node_edge_attr_sum / node_edge_count
    # Concatenate the original node features with the averaged edge attributes
    x_augmented = torch.cat([data.x, node_edge_attr_mean], dim=1)
    data.x = x_augmented
    return data


# Transformation for GDC
def GDC_transform(data):
    transform = GDC(
        diffusion_kwargs = dict(method='ppr', alpha=0.10, eps=0.03,),
        sparsification_kwargs = dict(method='threshold', avg_degree=32),
        exact = False,
        normalization_in = 'sym',
        normalization_out = 'sym',
    )
    data = transform(data)
    return data
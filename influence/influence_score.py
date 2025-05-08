# Author: Baskaran Sripathmanathan
# Link: https://openreview.net/profile?id=~Baskaran_Sripathmanathan1


import torch
import numpy as np
from tqdm import tqdm
# We use the older version of this function for its lower memory requirements
from torch.autograd.functional import jacobian
from torch_geometric.utils import k_hop_subgraph


# An adjustment to k_hop_subgraph to give all of the per-hop subsets:
def k_hop_subsets_rough(node_idx, num_hops, edge_index, num_nodes):
    col, row = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)
    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
    return subsets


# Takes k-hop subsets and removes any in previous hops:
def k_hop_subsets_exact(node_idx, num_hops, edge_index, num_nodes, device):
    subsets_init = k_hop_subsets_rough(
        node_idx, num_hops, edge_index, num_nodes=num_nodes
    )
    res = [subsets_init[0].tolist()]
    acc = set(res[0])
    for s in subsets_init[1:]:
        exact = set(s.tolist()) - acc
        acc = acc.union(exact)
        res.append(list(exact))
    return [torch.tensor(exact, device=device, dtype=edge_index.dtype) for exact in res]


# Very fast, assumes model is same precision as data.x
# don't compile!
def jacobian_l1(model, data, max_hops, node, device, vectorize=True):
    k_hop_nodes, new_edge_index, mapping, _ = k_hop_subgraph(
        node, max_hops, data.edge_index, relabel_nodes=True
    )
    # mapping is where our `node' ended up
    mapping = mapping[0]
    # Move everything to gpu:
    model = model.to(device)
    new_node_feat = data.x[k_hop_nodes].to(device)
    new_edge_index = new_edge_index.to(device)
    model_eval = lambda x: model(x, new_edge_index)[mapping]
    res_small = (
        jacobian(model_eval, new_node_feat, vectorize=vectorize).abs().sum(dim=(0, 2))
    )
    res = torch.zeros(data.num_nodes, dtype=res_small.dtype, device=res_small.device)
    res[k_hop_nodes.sort().values] = res_small
    del new_edge_index
    return res


# Divide jacobian values into hops
def jacobian_l1_agg_per_hop(model, data, max_hops, node, device, vectorize=True):
    if type(node) == torch.Tensor:
        node = node.item()
    vec = jacobian_l1(model, data, max_hops, node, device, vectorize=vectorize)
    hopsets = k_hop_subsets_exact(
        node, max_hops, data.edge_index, data.num_nodes, vec.device
    )
    return torch.tensor([vec[s].sum() for s in hopsets])


#Calculates n-hop influence aggregates; assumes that model and data are on the cuda device 
#and that data.x and model are the same type (e.g. model.half() and data.x.half().
#Returns a num_nodes x num_hops tensor.
def total_influence(model, data, max_hops, device, vectorize=True, num_samples=1):
    if num_samples < 1:
        num_samples = (data.num_nodes * num_samples) // 100
    else:
        num_samples = num_samples

    nodes = torch.randperm(data.num_nodes)[:num_samples]
    res = []
    res = torch.vstack(
        [jacobian_l1_agg_per_hop(model, data, max_hops, n, device, vectorize=vectorize) for n in tqdm(nodes)]
    )
    return res


# The influence-weighted receptive field R
def influence_weighted_receptive_field(T):
    breadth = np.mean(
        (T/T.sum(axis=0, keepdims=True)) @ np.arange(T.shape[0])
    )
    return breadth
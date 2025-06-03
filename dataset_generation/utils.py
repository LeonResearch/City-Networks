import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def rand_train_test_idx(label, train_prop=.1, valid_prop=.1, seed=0):
    """ randomly splits label into train/valid/test splits """
    n = label.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.RandomState(seed=seed).permutation(n))
    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    return train_indices, val_indices, test_indices


# %%%%%% Compute the Homophily Score %%%%%%
def compute_node_homophily(idx, labeled_nodes, edge_index, node_labels, labels, K, missing=False):
    # Extract K-hop neighborhood
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(labeled_nodes[idx].item(), K, edge_index, relabel_nodes=False)
    # Get the labels of nodes in the K-hop neighborhood
    neighborhood_labels = node_labels[sub_nodes]
    # Filter out nodes that are not labeled
    valid_neighborhood = neighborhood_labels != 0 if missing else torch.ones(neighborhood_labels.shape[0], dtype=bool)
    if valid_neighborhood.sum() == 0:
        return 0.0  # Skip this node if no valid neighbors with labels
    # Calculate the homophily score for this node
    same_label_count = (neighborhood_labels[valid_neighborhood] == labels[idx]).sum().float()
    total_label_count = valid_neighborhood.sum().float()
    homophily_score = same_label_count / total_label_count
    return homophily_score.item()


def compute_k_hop_homophily(node_labels, edge_index, K, node_mask=None, sampling_rate=0.1):
    # Filter out the un-labeled nodes
    labeled_nodes = (
        torch.nonzero(node_mask).squeeze() 
        if node_mask is not None 
        else torch.arange(node_labels.shape[0])
    )
    labeled_size = labeled_nodes.shape[0]

    # use a 10% sample if the labeled nodes are more than 1e5
    if labeled_size > 1e5:
        torch.manual_seed(0)
        indices = torch.randperm(labeled_size)[: min(int(labeled_size*sampling_rate), int(1e5)) ]  # Get k random indices
        labeled_nodes = labeled_nodes[indices]
    # Extract labels for labeled nodes (excluding "nan")
    labels = node_labels[labeled_nodes]
    # Use multiprocessing to compute homophily scores in parallel
    func = partial(
        compute_node_homophily, 
        labeled_nodes=labeled_nodes, 
        edge_index=edge_index, 
        node_labels=node_labels, 
        labels=labels, 
        K=K
    )
    homophily_scores = torch.zeros(len(labeled_nodes))
    print(f"%%%%%% Computing the homophily socre based on {labeled_nodes.shape[0]}/{labeled_size} nodes %%%%%%")
    for i in tqdm(range(labeled_nodes.shape[0])):
        homophily_scores[i] = func(i)
    # Compute the average homophily score
    avg_homophily_score = homophily_scores.mean()
    return avg_homophily_score.item()


def compute_k_hop_eccentricity(edge_index, edge_weight, k):
    num_nodes = torch.max(edge_index).item() + 1  # Total number of nodes
    sampled_nodes = list(range(num_nodes))

    edges = edge_index.T.tolist()
    weights = edge_weight.tolist()
    graph = nx.Graph()
    for (u, v), w in zip(edges, weights):
        graph.add_edge(u, v, weight=w)
    
    eccentricities = []
    for node in tqdm(sampled_nodes, desc=f"Computing eccentricity @ {k}-hop", unit="node"):
        # Extract k-hop subgraph
        sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(
            node, k, edge_index, relabel_nodes=True
        )
        # Convert k-hop subgraph to NetworkX format
        subgraph = graph.subgraph(sub_nodes.tolist())
        # Calculate eccentricity for the node within its k-hop subgraph
        ecc = nx.eccentricity(subgraph, v=node, weight='weight')
        eccentricities.append(ecc)
    return torch.tensor(eccentricities), torch.tensor(sampled_nodes)


def count_k_hop_labels(node_label, edge_index, k):
    k_hop_label_counts = torch.zeros(node_label.shape[0])
    for i in tqdm(range(node_label.shape[0])):
        k_hop_nodes = k_hop_subgraph(i, k, edge_index, relabel_nodes=False)[0]
        k_hop_label_counts[i] = node_label[k_hop_nodes].any(dim=0).sum()
    return k_hop_label_counts


def create_partition_labels(target, n_chunks):
    print(f"Creating labels by splitting data into {n_chunks} chunks")
    # First sort the target value of each node
    sorted_distances, sorted_indices = torch.sort(target)
    # get the size of each chunk
    n = len(target)
    chunk_size = n // n_chunks + 1
    labels = torch.zeros(n, dtype=torch.long)
    # assign labels to each chunk in ascending order
    chunk_indices = torch.arange(n_chunks).repeat_interleave(chunk_size)[:n]
    # match the labels with indices
    labels[sorted_indices] = chunk_indices
    return labels
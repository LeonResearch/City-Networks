import torch
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


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


def compute_k_hop_eccentricity(edge_index, edge_weight, k, node_samples=None, sample_rate=0.1):
    num_nodes = torch.max(edge_index).item() + 1  # Total number of nodes
    if sample_rate < 1: # Randomly sample nodes
        torch.manual_seed(0)
        sample_size = int(sample_rate * num_nodes)
        sampled_nodes = torch.randperm(num_nodes)[:sample_size].tolist()
    else:
        sampled_nodes = list(range(num_nodes))

    if node_samples is not None:
        sampled_nodes = node_samples

    edges = edge_index.T.tolist()
    weights = edge_weight.tolist()
    graph = nx.Graph()
    for (u, v), w in zip(edges, weights):
        graph.add_edge(u, v, weight=w)
    
    eccentricities = []
    for node in tqdm(sampled_nodes, desc="Computing eccentricity", unit="node"):
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


def haversine(coords1, coords2):
    """
    Calculate the great-circle distance between pairs of coordinates using the Haversine formula.
    Assumes coordinates are in [longitude, latitude] format.
    
    Args:
    - coords1 (torch.Tensor): Tensor of shape (N, 2), where each row is [longitude, latitude] in degrees.
    - coords2 (torch.Tensor): Tensor of shape (N, 2), where each row is [longitude, latitude] in degrees.
    
    Returns:
    - torch.Tensor: Distance between each pair of coordinates in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers

    # Convert degrees to radians manually
    deg_to_rad = torch.pi / 180.0
    lon1, lat1 = coords1[:, 0] * deg_to_rad, coords1[:, 1] * deg_to_rad  # Swap to [lat, lon]
    lon2, lat2 = coords2[:, 0] * deg_to_rad, coords2[:, 1] * deg_to_rad

    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply the Haversine formula
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Return the distance in kilometers
    return R * c


def compute_k_hop_euclidean(node_feat, edge_index, edge_weight, k, node_samples=None, sample_rate=0.1):
    if sample_rate < 1:
        torch.manual_seed(0)
        num_nodes = torch.max(edge_index).item() + 1  # Total number of nodes
        sample_size = int(sample_rate * num_nodes)
        sampled_nodes = torch.randperm(num_nodes)[:sample_size]  # Randomly sample nodes
    else:
        sampled_nodes = list(range(num_nodes))
    
    if node_samples is not None:
        sampled_nodes = node_samples
    
    euclidean = []
    for node in tqdm(sampled_nodes, desc="Computing euclidean distances", unit="node"):
        sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(
            node.item(), k, edge_index, relabel_nodes=False
        )
        coord1 = node_feat[node,:2].repeat([len(sub_nodes),1])
        coord2 = node_feat[sub_nodes, :2]

        dist = haversine(coord1, coord2)
        euclidean.append(dist.max().item())
    return euclidean, sampled_nodes
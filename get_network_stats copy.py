import torch
from torch_geometric.datasets import Planetoid, LRGBDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np

data_name = "ogbn-arxiv"

# Load specified dataset
if data_name in ["Cora", "Citeseer", "Pubmed"]:
    dataset = Planetoid(root=f'./data/{data_name}', name=data_name)
elif data_name in ["PascalVOC-SP", "COCO-SP"]:
    dataset = LRGBDataset(root=f'./data/{data_name}', name=data_name)
    dataset = dataset [:1000]
elif data_name in ["ogbn-arxiv"]:
    dataset = PygNodePropPredDataset(root=f'./data/{data_name}', name=data_name,)

def get_stats(data):
    graph = to_networkx(data, to_undirected=True)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    # Calculate average and standard deviation of node degrees
    degrees = [d for _, d in graph.degree()]
    avg_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    max_degree = np.max(degrees)
    print(f"max degree is {max_degree}")
    # average clustering coefficient
    avg_clustering_coef = nx.average_clustering(graph)
    # transitivity
    transitivity = nx.transitivity(graph)
    # Calculate graph diameter
    # Ensure graph is connected to compute the diameter
    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
    else:
        # For disconnected graphs, find the largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
    # Calculate homophily (example based on label agreement)
    labels = data.y.numpy()
    node_pairs = graph.edges()
    same_label_count = sum(1 for u, v in node_pairs if labels[u] == labels[v])
    homophily = same_label_count / len(node_pairs)
    return (
        num_nodes, 
        num_edges, 
        avg_degree, 
        std_degree, 
        max_degree,
        avg_clustering_coef, 
        transitivity,
        diameter,
        homophily,
    )


stats = {
    "num_nodes":[], 
    "num_edges":[], 
    "avg_degree":[], 
    "std_degree":[], 
    "max_degree":[],
    "avg_clustering_coef":[], 
    "transitivity":[],
    "diameter":[],
    "homophily":[],
}

for data in dataset:
    (
        num_nodes, 
        num_edges, 
        avg_degree, 
        std_degree, 
        max_degree,
        avg_clustering_coef, 
        transitivity,
        diameter,
        homophily
    ) = get_stats(data)

    stats["num_nodes"].append(num_nodes)
    stats["num_edges"].append(num_edges)
    stats["avg_degree"].append(avg_degree)
    stats["std_degree"].append(std_degree)
    stats["max_degree"].append(max_degree)
    stats["avg_clustering_coef"].append(avg_clustering_coef)
    stats["transitivity"].append(transitivity)
    stats["diameter"].append(diameter)
    stats["homophily"].append(homophily)

# Print results
print(f"Number of Nodes: {np.array(stats['num_nodes']).mean()}")
print(f"Number of Edges: {np.array(stats['num_edges']).mean()}")
print(f"Average Degree: {np.array(stats['avg_degree']).mean():.4f}")
print(f"Standard Deviation of Degree: {np.array(stats['std_degree']).mean():.4f}")
print(f"Max degree: {np.array(stats['max_degree']).mean():.4f}")
print(f"Average Clustering Coefficient: {np.array(stats['avg_clustering_coef']).mean():.4f}")
print(f"Transitivity: {np.array(stats['transitivity']).mean():.4f}")
print(f"Diameter: {np.array(stats['diameter']).mean()}")
print(f"Homophily: {np.array(stats['homophily']).mean():.4f}")

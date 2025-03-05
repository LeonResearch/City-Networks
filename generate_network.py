import os
import torch
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from places import generate_places
from generate_dataset import feature_engineering


# %%%%%% Setting up configurations %%%%%%
parser = argparse.ArgumentParser()
parser.add_argument('--place', type=str, default='paris')
parser.add_argument('--type', type=str, default='all')
parser.add_argument('--K_list', type=list[int], default=[16])
parser.add_argument('--n_bins_label', type=int, default=10)
parser.add_argument('--retain_all', type=bool, default=False)
parser.add_argument('--augment_node_attr', type=bool, default=True)

args = parser.parse_args()
print(args)
# 1. Load the Graph from OSMnX into a networkx Graph
#### 1.1 First specify all the configurations in this file
#['usa_main','usa_east_central','europe_west','uk','england','london','la','shanghai']
place_name = args.place 
data_dir = f'./data/{place_name}/'
places = generate_places(place_name)

plot = True
os.makedirs(data_dir, exist_ok=True)

start_time = time()
print(f"%%%%%% creating road network for {place_name} ... %%%%%%")
# use retain_all to keep all disconnected subgraphs (e.g. if your places aren't contiguous)
G = ox.graph_from_place(places, network_type=args.type, retain_all=args.retain_all)
print(f'finished in {(time() - start_time):.2f}s!')

start_time = time()
print(f"%%%%%% Gathering landuse info for {place_name} ... %%%%%%")
tags = {"landuse": True}  # Only landuse tags
land_use = ox.geometries_from_place(places, tags)
print(f'finished at {(time() - start_time):.2f}s!')

if plot:
    start_time = time()
    fig, ax = plt.subplots(1,1, figsize=(20, 20), gridspec_kw={'width_ratios': [1]})
    ox.plot_graph(G, node_size=0, edge_linewidth=1, edge_color='black', edge_alpha=0.5, show=False, ax=ax)
    plt.savefig(f"{data_dir}{place_name}_visual_map.jpg")
    plt.show()
    print(f'plotting finished in {(time() - start_time):.2f}s')

# Add an edge feature of speed limit on the road
G = ox.add_edge_speeds(G)

## 2. Some Exploratory Data Analysis
print(f"Keys from G dict: {G.__dict__.keys()}")

# ## 2.1 Node Features
nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
nodes_with_land_use = gpd.sjoin_nearest(nodes, land_use, how="inner") # get the landuse info
nodes = nodes.join(nodes_with_land_use.landuse, how="inner") 
nodes = nodes[~nodes.index.duplicated(keep="first")] # drop repeated index
nodes = nodes.drop(columns='geometry')
print("Description of nodes", nodes.describe())

try:
    print(nodes.street_count.value_counts(dropna=False))
    print(nodes.highway.value_counts(dropna=False))
    print(nodes.ref.value_counts(dropna=False))
    print(nodes.landuse.value_counts(dropna=False))
except:
    pass

# ## 2.2 Edge Features
edges = ox.graph_to_gdfs(G, nodes=False)
edges = edges.drop(columns='geometry')
edges["highway"] = edges["highway"].astype(str)

# ### 2.2.2 Numerical Edge Features
numerical_edge_feature_list = ["length", "speed_kph"]
for name in numerical_edge_feature_list:
    print(edges[name].describe(), '\n')

# ### 2.2.3 Categorical Edge Features
categorical_edge_feature_list = ["oneway", "lanes", "reversed", "junction", "tunnel", "service", "access"]
for name in categorical_edge_feature_list:
    try:
        print(edges[name].value_counts(dropna=False), '\n')
    except:
        pass

# %%%%%% Record Network Statistics %%%%%%
G_ = nx.Graph(G)

print(f"# Nodes: {G_.number_of_nodes()}, # Edges: {G_.number_of_edges()}")

node_degrees = torch.tensor([i[-1] for i in list(G_.degree())]).to(float)
avg_degree = node_degrees.mean()
std_degree = node_degrees.std()

avg_clustering_coef = nx.average_clustering(G_)
transitivity = nx.transitivity(G_)

def get_diameter_est(Graph):
    x_max_idx, x_min_idx = nodes.x.argmax(), nodes.x.argmin()
    y_max_idx, y_min_idx = nodes.y.argmax(), nodes.y.argmin()
    x_shortest_path = nx.shortest_path_length(Graph, source=nodes.iloc[x_max_idx].name, target=nodes.iloc[x_min_idx].name)
    y_shortest_path = nx.shortest_path_length(Graph, source=nodes.iloc[y_max_idx].name, target=nodes.iloc[y_min_idx].name)
    diameter_est = max(x_shortest_path, y_shortest_path)
    return diameter_est

try:
    diameter_est_directed = get_diameter_est(G_)
except: # if there happens to be no path between these two locations
    diameter_est_directed = None
    pass

print(f"Avg. degree: {avg_degree:.4f}")
print(f"Std. degree: {std_degree:.4f}")
print(f"Avg. clustering coef: {avg_clustering_coef:.4f}")
print(f"Transitivity: {transitivity:.4f}")
print(f"Diameter Estimate Directed: {diameter_est_directed}")
# %%%%%% End of Network Statistics %%%%%%   

# Feature Engineering
nodes = nodes.reset_index()
edges = edges.reset_index()

K_list=args.K_list
n_bins = args.n_bins_label
top_k_dict = {
    "oneway": 4,
    "lanes": 8, 
    "reversed": 4,
    "highway": 8,
    "landuse":8,
}

feature_engineering(args, data_dir, nodes, edges, K_list, n_bins, top_k_dict)


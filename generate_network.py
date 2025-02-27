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
parser.add_argument('--place', type=str, default=None)
parser.add_argument('--type', type=str, default='all')
parser.add_argument('--K_hop_estimation', type=int, default=16)
parser.add_argument('--n_bins_label', type=int, default=10)
parser.add_argument('--retain_all', type=bool, default=False)
parser.add_argument('--testing', type=bool, default=False)
parser.add_argument('--save_original_csv', type=bool, default=False)
parser.add_argument('--augment_node_attr', type=bool, default=True)

args = parser.parse_args()
print(args)
# 1. Load the Graph from OSMnX into a networkx Graph
#### 1.1 First specify all the configurations in this file

#place_name = 'usa_main'
#place_name = 'usa_east_central'
#place_name = 'europe_west'
#place_name = 'uk'
#place_name = 'england'
#place_name = 'london'
place_name = 'paris'
#place_name = 'la'
#place_name = 'shanghai'


if args.place is not None:
    place_name = args.place 

data_dir, places = generate_places(place_name)

if args.testing:
    place_name = 'testing'
    data_dir = './road_data/testing/'
    places = [
        #"London, England, UK"
        #"Utrecht, The Netherlands",
        #"New York State, USA",
        "Paris, France",
    ]

plot = True
save_and_load_G = False # Not recommended since saving the DataFrame takes a long time
save_type = 'csv'

savepath_G = data_dir + "G.graphml"
os.makedirs(data_dir, exist_ok=True)

if save_type == 'csv':
    savepath_node = data_dir + "nodes.csv"
    savepath_edge = data_dir + "edges.csv"
elif save_type == 'h5':
    savepath_node = data_dir + "nodes.h5"
    savepath_edge = data_dir + "edges.h5"

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

if save_and_load_G:
    start_time = time()
    print(f'now saving the network ...')
    ox.save_graphml(G, savepath_G)
    print(f'saving finished at {(time() - start_time):.2f}s')
    print(f'loading the network ...')
    G = ox.load_graphml(savepath_G)
    print(f'loading finished at {(time() - start_time):.2f}s')

print(f"# Nodes: {G.number_of_nodes()}, # Edges: {G.number_of_edges()}")

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
G_ = nx.DiGraph(G)

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

## Save the node and edge dataframes
if args.save_original_csv:
    start_time = time()
    print(f'Saving the node features and edge features as csv ...')
    if save_type == 'csv':
        nodes.to_csv(savepath_node)
        edges.to_csv(savepath_edge)
    elif save_type == 'h5':
        nodes.to_hdf(savepath_node, key='stage', mode='w')
        edges.to_hdf(savepath_edge, key='stage', mode='w')
    print(f'saving finished at {(time() - start_time):.2f}s')
else:
    nodes = nodes.reset_index()
    edges = edges.reset_index()

    K=args.K_hop_estimation
    n_bins = args.n_bins_label
    top_k_dict = {
        "oneway": 4,
        "lanes": 8, 
        "reversed": 4,
        "highway": 8,
        "landuse":8,
    }

    feature_engineering(args, data_dir, nodes, edges, K, n_bins, top_k_dict)


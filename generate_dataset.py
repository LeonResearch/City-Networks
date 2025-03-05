import numpy as np
import torch
from time import time
from torch_geometric.utils import to_undirected, remove_self_loops
from sklearn.preprocessing import OneHotEncoder

from utils import (
    rand_train_test_idx,
    compute_k_hop_homophily, 
    create_partition_labels,
    compute_k_hop_eccentricity,
)


# Feature Engineering
def feature_engineering(args, data_dir, node_df, edge_df, K_list, n_bins, top_k_dict):
    node_numerical_features = ["x", "y", "street_count"]
    node_categorical_features = ["landuse"]
    edge_numerical_features = ["length", "speed_kph"]
    edge_categorical_features = ["oneway", "lanes", "reversed", "highway"]

    # Relabel node index and edge index from 0
    relabel_map_dict = dict(zip(node_df.osmid, node_df.index))

    edge_df['u_relabeled'] = edge_df['u'].apply(lambda x: relabel_map_dict[x])
    edge_df['v_relabeled'] = edge_df['v'].apply(lambda x: relabel_map_dict[x])

    # Function to get top k categories in terms of value counts and then one-hot encode them
    def one_hot_top_k(df, k, column_name, node_label=False):
        df[column_name] = df[column_name].astype(str)
        df_value_count = df[column_name].value_counts(dropna=False)
        top_k = df_value_count.nlargest(k).index
        n_unique = len(df[column_name].unique().tolist())
        ratio_covered = df[column_name].value_counts()[:k].sum()/df[column_name].count()*100
        print(f"the largest {min(k, n_unique)} categories for '{column_name}' feature covers {ratio_covered:.1f}%, "
            f"they are \n {df_value_count.nlargest(k)} \n")
        
        df[column_name] = df[column_name].apply(lambda x: x if x in top_k else 'others')
        final_categories = df[column_name].value_counts().index.tolist()
        encoder = OneHotEncoder(categories=[final_categories], sparse_output=False)
        encoded = encoder.fit_transform(df[[column_name]])
        if 'nan' in final_categories:
            nan_index = final_categories.index('nan')
            label_ratio = encoded[:,nan_index].sum()/encoded.shape[0]
            print(f"the label ratio is: {(1-label_ratio)*100:.2f}%; the index of nan is: {nan_index}")
        else:
            print(f"No NA detected!")
            nan_index = None
        return encoded, df[column_name].unique().tolist(), nan_index

    edge_features_categorical = np.empty(shape=(edge_df.shape[0], 0))
    for item in edge_categorical_features:
        label_encoded, label_categories, _ = one_hot_top_k(edge_df, top_k_dict[item], item)
        edge_features_categorical = np.hstack((edge_features_categorical, label_encoded))
    print(f"In total, the edge categorical variables lead to {edge_features_categorical.shape[-1]} one-hot variables")
    print("shape of eadge_features_categorical", edge_features_categorical.shape)

    node_features_categorical = np.empty(shape=(node_df.shape[0], 0))
    for item in node_categorical_features:
        label_encoded, label_categories, _ = one_hot_top_k(node_df, top_k_dict[item], item)
        node_features_categorical = np.hstack((node_features_categorical, label_encoded))        
    print(f"In total, the node categorical variables lead to {node_features_categorical.shape[-1]} one-hot variables")
    print("shape of node_features_categorical", node_features_categorical.shape)

    # Transform to a PyG Data object
    node_features = torch.tensor(
        np.hstack(
            (node_df[node_numerical_features].values, node_features_categorical)
        ),
        dtype=torch.float32,
    )
    edge_features = torch.tensor(
        np.hstack(
            (edge_df[edge_numerical_features].values, edge_features_categorical)
        ),
        dtype=torch.float32
    )
    edge_indices = torch.tensor(edge_df[['u_relabeled', 'v_relabeled']].values, dtype=torch.long).T

    edge_indices, edge_features = to_undirected(edge_indices, edge_features, reduce="mean")
    edge_indices, edge_features = remove_self_loops(edge_indices, edge_features)

    # Save the node and edge features
    torch.save(node_features, f'{data_dir}node_features.pt')
    torch.save(edge_indices, f'{data_dir}edge_indices.pt')
    torch.save(edge_features, f'{data_dir}edge_features.pt')

    train_mask, valid_mask, test_mask = rand_train_test_idx(
        node_features, 
        train_prop=0.1, 
        valid_prop=0.1, 
        seed=0
    )

    torch.save(train_mask, f"{data_dir}train_mask.pt")
    torch.save(valid_mask, f"{data_dir}valid_mask.pt")
    torch.save(test_mask, f"{data_dir}test_mask.pt")

    if args.augment_node_attr:
        # Create empty tensor for sum and count
        node_edge_attr_sum = torch.zeros(node_features.shape[0], edge_features.shape[1])                          
        node_edge_count = torch.zeros(node_features.shape[0], 1)
        # sum the corresponding edge attr for each node
        node_edge_attr_sum.index_add_(0, edge_indices[1], edge_features)
        node_edge_count.index_add_(
            0,
            edge_indices[1],
            torch.ones_like(edge_indices[1], dtype=torch.float).view(-1, 1)
        )
        # Prevent division by zero for nodes with no neighbors if the graph is not connected
        node_edge_count[node_edge_count == 0] = 1.0
        # Compute the mean of the neighbouring edge attributes for each node
        node_edge_attr_mean = node_edge_attr_sum / node_edge_count
        # Concatenate the original node features with the averaged edge attributes
        x_augmented = torch.cat([node_features, node_edge_attr_mean], dim=1)
        torch.save(x_augmented, f'{data_dir}node_features_augmented.pt')

    # Create node labels by computing its K-hop eccentricity, 
    # i.e. the longest shortest path to other nodes within K hops 
    for K in K_list:
        start_time = time()
        eccentricities, _ = compute_k_hop_eccentricity(
            edge_index=edge_indices, 
            edge_weight=edge_features[:,0], # use road length as edge weight
            k=K,
        )
        node_labels = create_partition_labels(eccentricities, n_bins)
        print(f"eccentricity @ {K}-hop finished in {(time()-start_time):.1f}s")
        torch.save(node_labels, f'{data_dir}{n_bins}-chunk_{K}-hop_node_labels.pt')
        torch.save(eccentricities, f'{data_dir}{K}-hop_eccentricities.pt')
        print(node_labels.shape, node_features.shape, edge_features.shape, edge_indices.shape)

        # Compute K-hop homophily of the network
        homophily_score = compute_k_hop_homophily(node_labels, edge_indices, K=1)
        print(f"The node homophily score under {K}-hop eccentricty label is: {homophily_score:.2f}")
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import gzip

# Helper function to read a compressed CSV file
def read_csv_gz(path, **kwargs):
    with gzip.open(path, 'rt', newline='') as f:
        return pd.read_csv(f, **kwargs)

# Helper function to write a compressed CSV file
def write_csv_gz(dataframe, path):
    with gzip.open(path, 'wt', newline='') as f:
        dataframe.to_csv(f, index=False)

# This code works but make sure all relevant files are taken care of
# Maybe the code below would work

# num_edges = len(new_edges_df)
# num_edges_df = pd.DataFrame({'num_edges': [num_edges]})
# write_csv_gz(num_edges_df, '/path/to/raw/num-edge-list.csv.gz')  # Update with the correct path

# # Compute the degree of each node for the num-node-list.csv.gz file
# node_degrees = new_edges_df['source'].append(new_edges_df['target']).value_counts().sort_index()
# num_node_list_df = pd.DataFrame({'node_id': node_degrees.index, 'degree': node_degrees.values})
# write_csv_gz(num_node_list_df, '/path/to/raw/num-node-list.csv.gz')  # Update with the correct path

# print("num-edge-list and num-node-list have been updated.")



# # Step 1: Load the necessary files
# node_feat_path = 'arxiv/raw/node-feat.csv.gz'  # Update with the correct path
# edge_path = 'arxiv/raw/edge.csv.gz'  # Update with the correct path

# node_feat = read_csv_gz(node_feat_path, header=None)
# edges = read_csv_gz(edge_path, header=None, names=['source', 'target'])

# # Step 2: Perform PCA on the node features
# pca = PCA(n_components=100)
# node_feat_reduced = pd.DataFrame(pca.fit_transform(node_feat))

# # Step 3: Modify the edge list
# # Convert edge list to adjacency list for easier manipulation
# adj_list = edges.groupby('source')['target'].apply(list).to_dict()

# # Halve the neighbors if more than 1 neighbor
# for node, neighbors in adj_list.items():
#     if len(neighbors) > 1:
#         adj_list[node] = list(np.random.choice(neighbors, size=len(neighbors) // 2, replace=False))

# # Convert back to edge list format
# new_edges = [(node, neighbor) for node, neighbors in adj_list.items() for neighbor in neighbors]
# new_edges_df = pd.DataFrame(new_edges, columns=['source', 'target'])

# # Step 4: Save the modified node features and edge list back to compressed CSV files
# write_csv_gz(node_feat_reduced, 'arxiv/raw/node-feat.csv.gz')  # Update with the correct path
# write_csv_gz(new_edges_df, 'arxiv/raw/edge.csv.gz')  # Update with the correct path

# print("Modified data saved.")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import seaborn as sns
import random
import torch
import string
import re
from torch_geometric.loader import LinkNeighborLoader
import matplotlib.pyplot as plt
import os
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F 
from torch import Tensor
import tqdm
import pyg_lib

params = {'legend.fontsize': 'medium',
          'figure.figsize': (10, 8),
          'figure.dpi': 100,
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
plt.rcParams.update(params)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#read the dataframe from Kaggle 
movies_df = pd.read_csv("data/movies.csv",index_col='movieId')
ratings_df = pd.read_csv("data/ratings.csv")

print("The dataframes were imported successfully!")


# Split genres and convert into indicator variables:
genres = movies_df['genres'].str.get_dummies('|')
print(genres[["Action", "Adventure", "Drama", "Horror"]].head())
# Use genres as movie input features:
movie_feat = torch.from_numpy(genres.values).to(torch.float)
assert movie_feat.size() == (9742, 20)  # 20 genres in total.

# Create a mapping from unique user indices to range [0, num_user_nodes):
unique_user_id = ratings_df['userId'].unique()

unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedID': pd.RangeIndex(len(unique_user_id)),
})
print("Mapping of user IDs to consecutive values:")
print("==========================================")
print(unique_user_id.head())
print()


# Create a mapping from unique movie indices to range [0, num_movie_nodes):
unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedID': pd.RangeIndex(len(unique_movie_id)),
})
print("Mapping of movie IDs to consecutive values:")
print("===========================================")
print(unique_movie_id.head())


# Perform merge to obtain the edges from users and movies:
ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                            left_on='userId', right_on='userId', how='left')

print("ratings_user_id:")
print("===========================================")
print(ratings_user_id.head())

ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                            left_on='movieId', right_on='movieId', how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)
# With this, we are ready to construct our `edge_index` in COO format
# following PyG semantics:
edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
assert edge_index_user_to_movie.size() == (2, 100836)
print()
print("Final edge indices pointing from users to movies:")
print("=================================================")
print(edge_index_user_to_movie)




data = HeteroData()
# Save node indices:
data["user"].node_id = torch.arange(len(unique_user_id))
data["movie"].node_id = torch.arange(len(movies_df))
# Add the node features and edge indices:
data["movie"].x = movie_feat
data["user", "rates", "movie"].edge_index = edge_index_user_to_movie
# We also need to make sure to add the reverse edges from movies to users
# in order to let a GNN be able to pass messages in both directions.
# We can leverage the `T.ToUndirected()` transform for this from PyG:
data = T.ToUndirected()(data)


# For this, we first split the set of edges into
# training (80%), validation (10%), and testing edges (10%).
# Across the training edges, we use 70% of edges for message passing,
# and 30% of edges for supervision.
# We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
# Negative edges during training will be generated on-the-fly.
# We can leverage the `RandomLinkSplit()` transform for this from PyG:
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("user", "rates", "movie"),
    rev_edge_types=("movie", "rev_rates", "user"), 
)
train_data, val_data, test_data = transform(data)


# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:

# Define seed edges:
edge_label_index = train_data["user", "rates", "movie"].edge_label_index
edge_label = train_data["user", "rates", "movie"].edge_label
# train_loader = LinkNeighborLoader(
#     data=train_data,
#     num_neighbors=[20, 10],
#     neg_sampling_ratio=2.0,
#     edge_label_index=(("user", "rates", "movie"), edge_label_index),
#     edge_label=edge_label,
#     batch_size=128,
#     shuffle=True,
# )

import torch
import random
from torch.utils.data import DataLoader

# Function to sample neighbors and negative samples manually
class CustomLinkNeighborLoader:
    def __init__(self, data, num_neighbors, neg_sampling_ratio, edge_label_index, edge_label, batch_size, shuffle=True):
        self.data = data
        self.num_neighbors = num_neighbors
        self.neg_sampling_ratio = neg_sampling_ratio
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.edge_index = data[edge_label_index[0]].edge_index

    def _sample_neighbors(self, nodes):
        """
        Samples neighbors up to the specified number of neighbors per depth level.
        """
        sampled_neighbors = []
        for node in nodes:
            # Get all neighbors of this node
            neighbors = self.edge_index[1][self.edge_index[0] == node].tolist()
            # Sample `num_neighbors` randomly from the neighbors
            sampled_neighbors.extend(random.sample(neighbors, min(len(neighbors), self.num_neighbors[0])))
        return torch.tensor(sampled_neighbors)

    def _negative_sampling(self, pos_edges):
        """
        Generates negative samples by randomly pairing users with movies they have not rated.
        """
        users, movies = pos_edges[0], pos_edges[1]
        neg_edges = []
        all_movies = set(range(self.data['movie'].num_nodes))
        
        for user in users:
            rated_movies = set(movies[self.edge_index[0] == user].tolist())
            non_rated_movies = list(all_movies - rated_movies)
            sampled_negatives = random.sample(non_rated_movies, min(len(non_rated_movies), int(self.neg_sampling_ratio)))
            neg_edges.extend([(user, movie) for movie in sampled_negatives])

        return torch.tensor(neg_edges).T

    def _get_batches(self, pos_edges):
        """
        Splits edges into batches for training.
        """
        total_edges = len(pos_edges[0])  # Number of edges in the positive edge list
        indices = list(range(total_edges))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, total_edges, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            # Index into the first and second elements of the tuple separately
            batch_pos_edges = (pos_edges[0][batch_indices], pos_edges[1][batch_indices])

            # Sample negative edges for this batch
            batch_neg_edges = self._negative_sampling(batch_pos_edges)

            yield batch_pos_edges, batch_neg_edges


    def __iter__(self):
        """
        Yields batches of positive and negative edges for training.
        """
        pos_edges = self.edge_label_index
        return self._get_batches(pos_edges)


# Example of how to use the loader
def custom_link_neighbor_loader_example():
    # Assume 'train_data' and 'edge_label_index' are predefined
    loader = CustomLinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],  # Neighbor sampling at two layers
        neg_sampling_ratio=2.0,  # Negative sampling ratio
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True
    )

    for batch_pos_edges, batch_neg_edges in loader:
        print(f"Positive edges batch: {batch_pos_edges}")
        print(f"Negative edges batch: {batch_neg_edges}")

# Call the function to test the loader
train_loader = loader = CustomLinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],  # Neighbor sampling at two layers
        neg_sampling_ratio=2.0,  # Negative sampling ratio
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True
    )


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred
        
model = Model(hidden_channels=64)


for batch_pos_edges, batch_neg_edges in train_loader:
    print(f"Positive edges batch: {batch_pos_edges}")
    print(f"Negative edges batch: {batch_neg_edges}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 6):
    total_loss = total_examples = 0
    for sampled_data in (train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["user", "rates", "movie"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    
    


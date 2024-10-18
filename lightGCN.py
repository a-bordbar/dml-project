# © Alireza Bordbar
# bordbar@chalmers.se

# This script implements a LightGCN for recommendation on MovieLens 100k dataset.
# LighGCN is introduced in this paper: https://arxiv.org/abs/2002.02126


# ---------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch import Tensor, nn, optim
#from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul

# Importing the helper functions
from utils import *

# Read the "movies.csv" and "ratings.csv" and store them in two dataframes. 


# The "ratings.csv" contains "userId, movieId, and ratigs". 
# The "movies.csv" file contains "movieId" and the name of the movies.
movies_df  = pd.read_csv("./data/movies.csv", index_col="movieId")
ratings_df = pd.read_csv("./data/ratings.csv")

print("The dataset was imported successfully!")


edge_index, num_users, num_movies, movie_mapping, user_mapping = data_preprocessing(movies_df, ratings_df)

print("The data has been pre-processed!")

#Now we split the edges into train and test + val sets. 
# Extract the number of ratings
num_ratings = edge_index.shape[1]
rIdx = np.arange(num_ratings)


# 80% of the data is used for training; the rest is used for test and validation.
# For reproducibility, I set the random_state to 69.
train_index , test_val_index = train_test_split(rIdx, test_size=0.2, random_state = 69 )
val_index , test_index = train_test_split(test_val_index, test_size=0.5, random_state = 69 )

# Now that I have the training, validation, and test edge indices, I just need to create 
# a sparseTensor object that represents the adjacency matrix of the graph.
# The nodes do not change. However, we have three graphs with different sets of edges.
# This is done using "sparse_tensor_from_edgeIdx" functions, which can be found in "utils.py"
edgeIdx_train , edgeIdx_train_sparse = sparse_tensor_from_edgeIdx(train_index, edge_index , num_users, num_movies)
edgeIdx_val , edgeIdx_val_sparse = sparse_tensor_from_edgeIdx(train_index, edge_index , num_users, num_movies)
edgeIdx_test , edgeIdx_test_sparse = sparse_tensor_from_edgeIdx(train_index, edge_index , num_users, num_movies)


# Now I perform negative sampling on the training edges
edges = structured_negative_sampling(edgeIdx_train)
edges = torch.stack(edges, dim=0)



pass
    
# Alireza Bordbar 
# bordbar@chalmers.se
#https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
# Imports
import numpy as np 
import pandas as pd 
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score


# Read the dataset from the directory. The "movies.csv" file contains "movieId", the name of the movies, and their genres.
# The "ratings.csv" file contains "userId", "movieId", and the ratings from users for each movie.

movies_df = pd.read_csv("data/movies.csv",index_col='movieId')
ratings_df = pd.read_csv("data/ratings.csv")

print("The dataframes were imported successfully!")


# I use movie genres as features for movie nodes 
genres = movies_df['genres'].str.get_dummies('|')

features = torch.from_numpy(genres.values).to(torch.float)

# Make sure the dimension of the features is [#users + #movies, 20]

num_users = len(ratings_df['userId'].unique())
num_movies = len(ratings_df['movieId'].unique())



features = torch.from_numpy(genres.values).to(torch.float)

# Create a mapping from unique user indices to range [0, num_user_nodes):
unique_user_id = ratings_df['userId'].unique()

unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedID': pd.RangeIndex(len(unique_user_id)),
})


# Create a mapping from unique movie indices to range [0, num_movie_nodes):
unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedID': pd.RangeIndex(len(unique_movie_id)),
})


ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                            left_on='userId', right_on='userId', how='left')

ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                            left_on='movieId', right_on='movieId', how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)
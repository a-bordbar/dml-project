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





def data_preprocessing(movie_dataframe, rating_dataframe):
    '''
    inputs:
      movie_dataframe: the dataframe that contains the "movieId" variable and the name of the movies.
      rating_dataframe: the dataframe that contains the "movieId", "userId", and all the ratings. 
    
    
    outputs: 
      edge_index (Tensor):  the indices of the edges in the adjacency matrix.
      num_users (int): the number of the users.
      num_movies (int): the number of movies.
      user_mapping(dataframe): the dataframe that maps the userId in the original ratings_df 
                                to continuous indices from 0.
    '''
    
    
    # First, we map the indices of the movies from 0 to num_movies -1. 
    movie_mapping = {idx: i for i, idx in enumerate(movie_dataframe.index.unique())}

    
    # Then, we do the same thing for users (one of the outputs)
    user_mapping = {idx : i for i, idx in enumerate(rating_dataframe.index.unique())}
    
    # Then, extract the number of users and movies.
    num_users , num_movies = len(rating_dataframe.index.unique()), len(movie_dataframe.index.unique())
    
    
    # For each rating in "rating_dataframe", extract the corresponding user and movie.
    # store them in seperate lists
    users = [user_mapping[idx] for idx in rating_dataframe['userId']]
    movies = [movie_mapping[idx] for idx in rating_dataframe['movieId']]
    
    
    # The next step is crucial: I decide that any rating that is >= 3 is considered a positive
    # interaction between the corresponding user and movie.
    
    # recommend_bool has shape [100836, ]
    positive_interaction = torch.from_numpy(rating_dataframe['rating'].values).view(-1, 1).to(torch.long) >= 3
    
    
    # In the next step, I make a graph corersponding to different users and movies that 
    # have interacted positively. 
    
    edge_index = [[], []]
    for i in range(positive_interaction.shape[0]):
        if positive_interaction[i]:   # We filter out the negative interactions here.
            edge_index[0].append(users[i])
            edge_index[1].append(movies[i])
    
    edge_index = torch.tensor(edge_index)
    return edge_index, num_users, num_movies, movie_mapping, user_mapping



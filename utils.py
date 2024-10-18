import torch
from torch import Tensor, nn, optim
#from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm






def data_preprocessing(movie_dataframe, rating_dataframe):
    
    
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


def sparse_tensor_from_edgeIdx(edge_index_train, edge_index, num_users, num_movies):
    edgeIdx_new = edge_index [:, edge_index_train]
    num_nodes = num_users + num_movies
    sparse_edgeIdx = SparseTensor(row = edgeIdx_new[0],
                                  col = edgeIdx_new[1],
                                  sparse_sizes= (num_nodes, num_nodes)
                                  )
    return edgeIdx_new, sparse_edgeIdx
  

def mini_batch_sample(batch_size, adjacency_matrix):
    
    edges = structured_negative_sampling(adjacency_matrix)
    edges = torch.stack(edges, dim=0)
    indices = torch.randperm(edges.shape[1])[:batch_size]
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices


class LightGCN(MessagePassing):
    # Refer to https://arxiv.org/abs/2002.02126 for the paper 
    def __init__(self, num_users, num_movies, hidden_dim, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.movie_embedding = nn.Embedding(self.num_movies, self.hidden_dim)
        
        
        # Change this in case of not converging
        nn.init.normal_(self.user_embedding.weight, std=1.0)
        nn.init.normal_(self.movie_embedding.weight, std=1.0)

    def forward(self, edgeIdx):
        edgeIdx_norm = gcn_norm(edgeIdx , False) #Applies the GCN normalization from the `"Semi-supervised Classification
                                                 #with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
                                                 #paper (functional name: :obj:`gcn_norm`).

                                                 # math::
                                                 # \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
                                                 #\mathbf{\hat{D}}^{-1/2}

                                                 #where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.

        #concatenate the embeddings of users and movies
        embed_cat = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embed_cat_list = [embed_cat]
        
        for i in range(self.num_layers):
            embed_cat = self.propagate(edgeIdx_norm, x=embed_cat)
            embed_cat_list.append(embed_cat)
        
        embed_cat_list = torch.stack(embed_cat_list , dim=1)
        embed_cat_output= torch.mean(embed_cat_list , dim=1 )
        
        user_embedding_output, movie_embedding_output = torch.split(embed_cat_output, [[self.num_users, self.num_movies]])
        return user_embedding_output, user_embedding_output.weight, movie_embedding_output, movie_embedding_output.weight
    
    
    def message(self, x):
        return x

    def propagate(self, edge_index, x):
        x = self.message_and_aggregate(edge_index, x)
        return x

    def message_and_aggregate(self, edge_index, x):
        return matmul(edge_index, x)



def bpr(user_embedding, user_embedding_initial, positive_embedding, 
        positive_embedding_initial, negative_embedding, negative_embedding_initial, reg_coef):
    
    positive_score = torch.sum(user_embedding * positive_embedding, dim=1)
    negative_score = torch.sum(user_embedding * negative_embedding, dim=1)
    loss  = -torch.log(torch.sigmoid(positive_score - negative_score))
    loss = torch.mean(loss) + reg_coef * \
    (torch.norm(user_embedding_initial) + torch.norm(positive_embedding_initial) + torch.norm(negative_embedding_initial))
    return loss 
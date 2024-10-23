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
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv, GATConv





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
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, std=0.1)

    def forward(self, edgeIdx):
        edgeIdx_norm = gcn_norm(edgeIdx , False) #Applies the GCN normalization from the `"Semi-supervised Classification
                                                 #with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
                                                 #paper (functional name: :obj:`gcn_norm`).

                                                 # math::
                                                 # \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
                                                 #\mathbf{\hat{D}}^{-1/2}

                                                 #where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.

        #concatenate the embeddings of users and movies
        embed_cat = torch.cat([self.user_embedding.weight, self.movie_embedding.weight])
        embed_cat_list = [embed_cat]
        
        for i in range(self.num_layers):
            embed_cat = self.propagate(edgeIdx_norm, x=embed_cat)
            embed_cat_list.append(embed_cat)
        
        embed_cat_list = torch.stack(embed_cat_list , dim=1)
        embed_cat_output= torch.mean(embed_cat_list , dim=1 )
        
        user_embedding_output, movie_embedding_output = torch.split(embed_cat_output, [self.num_users, self.num_movies])
        return user_embedding_output, self.user_embedding.weight, movie_embedding_output, self.movie_embedding.weight
    
    
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


class GAT(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, num_layers):
        super(GAT, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layers for users and items
        self.users_emb = nn.Embedding(self.num_users, hidden_dim)
        self.items_emb = nn.Embedding(self.num_items, hidden_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        # Define GraphSAGE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=True))
            
            

class GraphSAGEModel(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, num_layers):
        super(GraphSAGEModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layers for users and items
        self.users_emb = nn.Embedding(self.num_users, hidden_dim)
        self.items_emb = nn.Embedding(self.num_items, hidden_dim)

        nn.init.normal_(self.users_emb.weight, std=0.01)
        nn.init.normal_(self.items_emb.weight, std=0.01)

        # Define GraphSAGE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, edge_index):
        x_user = self.users_emb.weight
        x_item = self.items_emb.weight
        
        # Concatenate user and item embeddings
        x = torch.cat([x_user, x_item], dim=0)

        # Perform neighborhood aggregation using SAGEConv layers
        for conv in self.convs:
            #x = F.relu(conv(x, edge_index))  #for SAGEConv
            x = conv(x, edge_index)  #For GCNConv
        
        # Separate back into user and item embeddings
        users_emb, items_emb = x[:self.num_users], x[self.num_users:]
        
        return users_emb, self.users_emb.weight, items_emb, self.items_emb.weight






def precision_recall(GNN_model, edgeIdx, sparse_edgeIdx, maskIdx, k, reg_coef):
    '''
    This function calculates the performance metrics for the model:
    Precision@k 
    Recall@k
    '''
    
    
    #First, I get the user and movie embeddings from the model.
    users_emb, users_emb_0, items_emb, items_emb_0 = GNN_model.forward(sparse_edgeIdx)
    edges = structured_negative_sampling(edgeIdx, contains_neg_self_loops=False)
    
    user_indices, pos_indices, neg_indices = edges[0], edges[1], edges[2]
    users_emb, users_emb_0 = users_emb[user_indices], users_emb_0[user_indices]
    pos_emb, pos_emb_0 = items_emb[pos_indices], items_emb_0[pos_indices]
    neg_emb, neg_emb_0 = items_emb[neg_indices], items_emb_0[neg_indices]

    loss = bpr(users_emb, users_emb_0, pos_emb, pos_emb_0,
                    neg_emb, neg_emb_0, reg_coef).item()
    
    user_embedding_weight = GNN_model.user_embedding.weight
    movie_embedding_weight = GNN_model.movie_embedding.weight
    
    #Calculate the score by dot product 
    scores = torch.matmul(user_embedding_weight, movie_embedding_weight.T)
    
    
    for idx in maskIdx:
        user_positive_movies = retrieve_positive_movies(idx)
        masked_users = []
        masked_movies = []
        for user, movies in user_positive_movies.items():
            masked_users.extend([user] * len(movies))
            masked_movies.extend(movies)
        scores[masked_users , masked_movies] = float("-inf")
    
    _, top_k_movies=torch.topk(scores, k=k)
    
    #Compare the rating with the actual rating 
    users = edgeIdx[0].unique()
    users_actual_positive_movies = retrieve_positive_movies(edgeIdx)
    
    actual_rating = [users_actual_positive_movies[user.item()] for user in users]
    predicted_rating = []
    
    for user in users:
        movies= users_actual_positive_movies[user.item()]
        rating = list (map(lambda x:x in movies, top_k_movies[user]))
        predicted_rating.append(rating)
    
    #Convert the predicted_rating list to a Tensor
    predicted_rating = torch.Tensor(np.array(predicted_rating).astype("float"))
    
    #Count the number of correct predictions 
    correct_prediction_count = torch.sum(predicted_rating, dim = -1)
    
    #Count the number of liked items by each user (actual)
    actual_positive_count = torch.Tensor([len(actual_rating[i]) for i in range(len(actual_rating))])
    
    #Calculate precision and recall 
    precision = torch.mean(correct_prediction_count) / k
    recall = torch.mean(correct_prediction_count/actual_positive_count)
    
    return precision , recall , loss
    
    
    
    
def retrieve_positive_movies(edgeIdx):
    
    positive_movies = {}
    for i in range(edgeIdx.shape[1]):
        user = edgeIdx[0][i].item()
        movie = edgeIdx[1][i].item()
        if user not in positive_movies:
            positive_movies[user] = []
        positive_movies[user].append(movie)
    return positive_movies






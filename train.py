

# This script implements a GNN for recommendation on MovieLens dataset.


# ---------------
#benchmark : https://github.com/lxbanov/recsys-movielens100k
from torch import Tensor, nn, optim
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
# Importing the helper functions
from utils import *
import torch.nn.functional as F

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
train_index , test_val_index = train_test_split(rIdx, test_size=0.2, random_state = 42 )
val_index , test_index = train_test_split(test_val_index, test_size=0.5, random_state = 42 )

# Now that I have the training, validation, and test edge indices, I just need to create 
# a sparseTensor object that represents the adjacency matrix of the graph.
# The nodes do not change. However, we have three graphs with different sets of edges.
# This is done using "sparse_tensor_from_edgeIdx" functions, which can be found in "utils.py"
edgeIdx_train , edgeIdx_train_sparse = sparse_tensor_from_edgeIdx(train_index, edge_index , num_users, num_movies)
edgeIdx_val , edgeIdx_val_sparse = sparse_tensor_from_edgeIdx(val_index, edge_index , num_users, num_movies)
edgeIdx_test , edgeIdx_test_sparse = sparse_tensor_from_edgeIdx(test_index, edge_index , num_users, num_movies)


# Now I perform negative sampling on the training edges
edges = structured_negative_sampling(edgeIdx_train)
edges = torch.stack(edges, dim=0)

#Here, I set up the simulation parameters
#----   ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    
batch_size = 256
num_epochs = 50
epoch_iterations = 200
learning_rate = 1e-4  #Set this to 1e-4 for SageCONV
exponentional_decay = 0.9
k = 20 #for calculating "top-k" performance 
regularization_factor = 1e-6 # for BPR loss
hidden_dimension = 64 # Hidden dimension of the embedding layer 
num_layers = 2
#----   ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    

#Set up the device 
device = torch.device("cude" if torch.cuda.is_available() else "cpu")

# Instantiate the model 
# Choose one Model structure 
#model = LightGCN(num_users , num_movies, hidden_dimension, num_layers)
#model = GraphSAGEModel(num_users , num_movies, hidden_dimension, num_layers)
model = GCN(num_users , num_movies, hidden_dimension, num_layers)


# Transfer the model and the data to the device 
model.to(device)

edge_index = edge_index.to(device)
edgeIdx_train = edgeIdx_train.to(device)
edgeIdx_train_sparse = edgeIdx_train_sparse.to(device)

edgeIdx_val = edgeIdx_val.to(device)
edgeIdx_val_sparse = edgeIdx_val_sparse.to(device)
# Put the model in train mode 


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#set up learning rate decay 
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = exponentional_decay)

# I store the train and validation losses in a list in order to plot them later

train_losses = []
val_losses = []

model.train()
for epoch in range(num_epochs):
    for iteration in range(epoch_iterations):
        # Forward pass 
        users_emb, users_emb_0, items_emb, items_emb_0 = \
        model.forward(edgeIdx_train_sparse)
        
        # Create mini-batches
        user_indices, pos_indices, neg_indices = \
        mini_batch_sample(batch_size, edgeIdx_train)
        
        user_indices = user_indices.to(device)
        pos_indices = pos_indices.to(device)
        neg_indices = neg_indices.to(device)
    
        users_emb, users_emb_0 = users_emb[user_indices], users_emb_0[user_indices]
        pos_emb, pos_emb_0 = items_emb[pos_indices], items_emb_0[pos_indices]
        neg_emb, neg_emb_0 = items_emb[neg_indices], items_emb_0[neg_indices]
        
        
         # loss computation
        loss = bpr(users_emb, users_emb_0, 
                    pos_emb, pos_emb_0,
                    neg_emb, neg_emb_0,
                    regularization_factor)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
    # val_loss, recall, precision = evaluation(model, edgeIdx_val, 
    #                                     edgeIdx_val_sparse, 
    #                                     [edgeIdx_train], 
    #                                     k,
    #                                     regularization_factor)
    
    precision , recall , val_loss = precision_recall(model, edgeIdx_val, 
                                        edgeIdx_val_sparse, 
                                        [edgeIdx_train], 
                                        k,
                                        regularization_factor)
    
    
    print('Epoch {:d}: train_loss: {:.4f}, val_loss: {:.4f}, recall: {:.4f}, precision: {:.4f}'\
    .format(epoch, loss, val_loss, recall, precision))
    train_losses.append(loss.item())
    val_losses.append(val_loss)
    scheduler.step()


#Test set evaluation

model.eval()
test_sparse_edge_index = edgeIdx_test_sparse.to(device)
test_edge_index = edgeIdx_test.to(device)
precision_test , recall_test , loss_test = precision_recall(model, 
                test_edge_index, 
                test_sparse_edge_index, 
                [edgeIdx_train, edgeIdx_val],
                k,
                regularization_factor)
print('Test loss: {:.4f}, Test recall: {:.4f}, Test precision: {:.4f}'\
        .format(loss_test, recall_test, precision_test))


#Save the outputs to a file.
output_files = {"num_epoch": num_epochs , "train_losses": train_losses , "val_losses":val_losses, "test_loss":loss_test,
                "test_precisioin" : precision_test, "test_recall":recall_test}
torch.save(output_files, "./outputs/GCN.pt")


# #Load the data for comparison and print them out
# monitor = True 
# if monitor:
#     loaded_data_LightGCN = torch.load("./outputs/LightGCN.pt")
#     loaded_data_GCN = torch.load("./outputs/GCN.pt")
#     loaded_data_SAGE = torch.load("./outputs/GraphSAGE.pt")
    
    
def predict(user_id, topK):
    '''
    Make top K recommendations to the user
    '''
    # read movie and uesr info
    model.eval()
    df  = pd.read_csv("./data/movies.csv", index_col="movieId")
    ratings_df = pd.read_csv("./data/ratings.csv")
    movie_titles = pd.Series(df.title.values, index=df.index).to_dict()
    movie_genres = pd.Series(df.genres.values, index=df.index).to_dict()
    pos_items = retrieve_positive_movies(edge_index)
    user = user_mapping[user_id]
    user_emb = model.users_emb.weight[user]
    scores = model.items_emb.weight @ user_emb

    values, indices = torch.topk(scores, k=len(pos_items[user]) + topK)

    movies = [index.cpu().item() for index in indices if index in pos_items[user]]
    topk_movies = movies[:topK] 
    movie_ids = [list(movie_mapping.keys())[list(movie_mapping.values())\
                                            .index(movie)] for movie in movies]
    titles = [movie_titles[id] for id in movie_ids]
    genres = [movie_genres[id] for id in movie_ids]

    print("User {:d} liked these movies:".format(user_id))
    for i in range(topK):
        print("{:s}, {:s} ".format(titles[i], genres[i]))

    print('====================================================================')

    movies = [index.cpu().item() for index in indices if index not in pos_items[user]]
    topk_movies = movies[:topK] 
    movie_ids = [list(movie_mapping.keys())[list(movie_mapping.values())\
    .index(movie)] for movie in movies]
    titles = [movie_titles[id] for id in movie_ids]
    genres = [movie_genres[id] for id in movie_ids]

    print("Here are the movies that we think the user will enjoy:")
    for i in range(topK):
        print("{:s}, {:s} ".format(titles[i], genres[i]))



predict(69, 5)
torch.save(model.state_dict(), "./GCN_saved.pt")
pass
    
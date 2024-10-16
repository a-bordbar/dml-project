# Alireza Bordbar 
# bordbar@chalmers.se
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


# Load the data (MovieLens 100k dataset)
movies_df = pd.read_csv("data/movies.csv", index_col='movieId')  #This csv file contains movieId, titles of the movies, and genre
ratings_df = pd.read_csv("data/ratings.csv")  #This csv file contains userId, movieId, and rating


#Here, I create user and movie mappings 
user_ids = ratings_df["userId"].unique()
movie_ids = ratings_df["movieId"].unique()

#create a dictionary that maps each user_id to an index. Bascially, we assign each user and movie a node index
user_to_idx = {user_id : idx for idx, user_id in enumerate(user_ids)}
# do the same thing for movies. The only difference is that we don't want the movies and user ids to overlap
movie_to_idx = {movie_id : idx +len(user_ids) for idx, movie_id in enumerate(movie_ids)}
 
 
# Map user and movie IDs to indices
ratings_df["userId"] = ratings_df["userId"].map(user_to_idx) #Apply the mapping defined above
ratings_df["movieId"] = ratings_df["movieId"].map(movie_to_idx) #Apply the mapping defined above


#Create edge indices 
edge_index = torch.tensor(ratings_df[['userId' , 'movieId']].values.T , dtype= torch.long)
edge_index = edge_index.contiguous()






#Each node in the GNN has a feature. Let us apply those 
num_users = len(user_to_idx)
num_movies = len(movie_to_idx) - len(user_to_idx)


user_features = torch.ones(num_users)
movie_features = torch.ones(num_movies)


# This feature embedding is wrong!!!
x = torch.cat([user_features, movie_features], dim = 0) #x has the shape [num_users + num_movies, feature_dim].
x = torch.unsqueeze(x, -1)

#Now, I create a graph data object
data = Data(x = x , edge_index= edge_index)

train_size = int(0.8 * len(ratings_df))
train_data = Data(x=data.x , edge_index= data.edge_index[: , :train_size])
test_data = Data(x=data.x , edge_index= data.edge_index[: , train_size:])


#Define the GNN model 
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize model, optimizer, etc.
model = GNNModel(in_channels=train_data.x.shape[1], hidden_channels=16, out_channels=1)


# Initialize model, optimizer, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(100):  # 100 epochs
    model.train()
    optimizer.zero_grad()                   # Clear gradients
    out = model(train_data.x, train_data.edge_index)  # Forward pass
    target = ratings_df['rating'].values[:train_size]  # Target ratings
    loss = loss_fn(out.flatten(), torch.tensor(target, dtype=torch.float32))  # Calculate loss
    loss.backward()                          # Backpropagation
    optimizer.step()                         # Update weights

    if epoch % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch {epoch}, Loss: {loss.item()}')


pass
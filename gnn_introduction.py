# Alireza Bordbar
# bordbar@chalmers.se


import os
import collections
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.sparse as sp
import os 
import torch 
import networkx as nx
from torch_geometric.utils import to_networkx


# I use PyTorch Geometric to build GNNs

import torch_geometric
from torch_geometric.datasets import MovieLens      #Import the MovieLens dataset
from torch_geometric.transforms import NormalizeFeatures    




#Make a directory for the data 
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)


#Put the Cora dataset in a pandas frame
dataset = MovieLens(root= data_dir)

data = dataset[0]  #This is global storage object. 

#Data Exploration
print("All attributes and methods:", data.__dir__()) #This prints out all the attributes and methods of the dataset

#Let us for example explore the number of nodes and edges in the datset 
print("The number of nodes in the datset: {}".format(data.num_nodes))
print("The number of edges in the datset: {}".format(data.num_edges))   #Because the graph is bi-directional, each edge is counted twice.

print(data) 

#After inspecting the dataset, we see that it is a bipartite graph, meaning one set of nodes represent the 
#users and another set of nodes represetn the movies
#In order to access the edge index, we need to specify the relation 

#There are different types of relations in the dataset. They can be accessed with 

print("Different relations: \n", data.edge_types)


edge_index = data ["user", "rates", "movie"].edge_index

print("User to movie edge index:\n", edge_index)


#Now, I create a bipartite representation of the dataset. For that, I represent movies and users 
#as two different types of nodes using the "networkx" package

#Initialize the bipartite graph 
B = nx.Graph()


# Get the user and movie nodes

user_nodes = edge_index[0].tolist()
movie_nodes = edge_index[1].tolist()


# Extract the number of user and movie nodes 

num_users = data["user"].num_nodes
num_movies = data["movie"].num_nodes

B.add_nodes_from(range(num_users), bipartite = 0) # Here, I put the users in set 0
B.add_nodes_from(range(num_users, num_users + num_movies), bipartite=1)  # I put movies in set 1


#Now, I add edges to the graph
edges = [(user, num_users + movie) for user, movie in zip(user_nodes, movie_nodes)]

# Pass this list to the "add_edges_from" attribute
B.add_edges_from(edges)

print("The number of nodes in the graph: {}".format(B.number_of_nodes()))
print("The number of edges in the graph: {}".format(B.number_of_edges()))


# Now the fun part begins! Let's visualize the bipartite graph
# In order to draw a graph using the "networkx" package, I need to define the custom position of the nodes

pos = {}

pos.update((n , (1, i))  for i, n in enumerate(range(num_users)))  # (n, (x,y)) means the node n is positioned at (x,y)
pos.update((n , (2, i-num_users))  for i, n in enumerate(range(num_users, num_users + num_movies)))


# #DEBUG
# missing_nodes = [n for n in B.nodes() if n not in pos]
# if missing_nodes:
#     print(f"Missing positions for nodes: {missing_nodes}")
# else:
#     print("All nodes have positions assigned.")
# #DEBUG END

plt.figure(figsize = (12, 8))
plt.title("A very crude plot of the graph")
nx.draw(B , pos= pos , with_labels= False , node_size = 20, node_color="blue", edge_color = "gray")
plt.show()

# # Here, I make the plot a bit cleaner. '
# # First, I'm going to use the spring layout.
# # WARNING: This will take a long time to run. Also, the plot is still cluttered
# # SOLUTION: Limit the number of users and movies displayed

# pos_spring = nx.spring_layout(B,k= 0.5)
# plt.figure(figsize=(12,8))

# # I assign diffrent colors to users and movies 
# user_color = "blue"
# movie_color = "orange"

# nx.draw_networkx_nodes(B, pos_spring , nodelist=range(num_users) , node_color=user_color,
#                        label = "Users", node_size= 20)

# nx.draw_networkx_nodes(B, pos_spring , nodelist=range(num_users, num_users + num_movies) , node_color=movie_color,
#                        label = "Movies", node_size= 20)

# nx.draw_networkx_edges(B, pos_spring, alpha=0.5)
# plt.title('Bipartite Graph: Users and Movies (from MovieLens)')
# plt.legend()
# plt.axis("off")
# plt.show()




# # Here, I limit the number of users and movies displayed.

# sample_size_users = 50  # Number of users to sample
# sample_size_movies = 50  # Number of movies to sample

# sample_users = set(range(sample_size_users))  # Sample first 50 users
# sample_movies = set(range(num_users, num_users + sample_size_movies))  # Sample first 50 movies

# B_sample = nx.Graph()
# B_sample.add_nodes_from(sample_users , bipartite = 0)
# B_sample.add_nodes_from(sample_movies , bipartite = 1)

# #Add the new edges 
# edges_sampled = [(user, num_users + movie) for user, movie in zip(user_nodes, movie_nodes)
#                  if user in sample_users and num_users + movie in sample_movies]

# B_sample.add_edges_from(edges_sampled)

# # Draw the sampled bipartite graph

# plt.figure(figsize=(12, 8))
# user_color = "blue"
# movie_color = "orange"
# pos_sample = nx.spring_layout(B_sample)
# nx.draw_networkx_nodes(B_sample, pos_sample, nodelist=sample_users, node_color=user_color, label='Users', node_size=30)
# nx.draw_networkx_nodes(B_sample, pos_sample, nodelist=sample_movies, node_color=movie_color, label='Movies', node_size=30)
# nx.draw_networkx_edges(B_sample, pos_sample, alpha=0.5)

# plt.title('Sampled Bipartite Graph: Users and Movies (from MovieLens)')
# plt.legend()
# plt.axis('off')
# plt.show()


# Step 1: Sample edges directly (instead of sampling users/movies separately)
num_edges = len(user_nodes)
sample_size_edges = 50  # Number of edges to sample
edge_sample_indices = torch.randperm(num_edges)[:sample_size_edges]  # Randomly sample 100 edges

# Get the sampled user and movie nodes from the sampled edges
sampled_user_nodes = [user_nodes[i] for i in edge_sample_indices]
sampled_movie_nodes = [movie_nodes[i] for i in edge_sample_indices]

# Step 2: Add the sampled edges to a new bipartite graph
B_sample = nx.Graph()

# Add user and movie nodes based on sampled edges
num_users = data['user'].num_nodes
B_sample.add_nodes_from(sampled_user_nodes, bipartite=0)  # Add users to set 0
B_sample.add_nodes_from([num_users + movie for movie in sampled_movie_nodes], bipartite=1)  # Add movies to set 1

# Add the edges from the sampled users and movies
sampled_edges = [(user, num_users + movie) for user, movie in zip(sampled_user_nodes, sampled_movie_nodes)]
B_sample.add_edges_from(sampled_edges)

# Step 3: Verify that the sampled graph has nodes and edges
print(f"Number of nodes in sampled graph: {B_sample.number_of_nodes()}")
print(f"Number of edges in sampled graph: {B_sample.number_of_edges()}")

# Step 4: Assign positions for visualization using spring layout
pos_sample = nx.spring_layout(B_sample, k= 0.5)  #Use this for spring layout

# pos_sample = {}  # Use this for a simple layout
# pos_sample.update((n, (1, i)) for i, n in enumerate(sampled_user_nodes))  # Users on one side (x = 1)
# pos_sample.update((n, (2, i)) for i, n in enumerate([num_users + movie for movie in sampled_movie_nodes]))  # Movies on the other side (x = 2)


# Step 5: Draw the sampled bipartite graph with distinct colors for users and movies
plt.figure(figsize=(10, 6))

# Separate users and movies for coloring
nx.draw_networkx_nodes(B_sample, pos_sample, nodelist=sampled_user_nodes, node_color='blue', label='Users', node_size=10)
nx.draw_networkx_nodes(B_sample, pos_sample, nodelist=[num_users + movie for movie in sampled_movie_nodes], node_color='orange', label='Movies', node_size=10)

# Draw edges
nx.draw_networkx_edges(B_sample, pos_sample, alpha=0.5)

# Add legend and display the graph
plt.legend(['Users', 'Movies'])
plt.title('Sampled Bipartite Graph: Users and Movies (from MovieLens)')
plt.axis('off')
plt.show()
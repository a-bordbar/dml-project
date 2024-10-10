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
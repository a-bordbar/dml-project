# Alireza Bordbar
# bordbar@chalmers. se


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


# I use PyTorch Geometric to build GNNs

import torch_geometric
from torch_geometric.datasets import Planetoid      #Import the Planetoid Cora dataset
from torch_geometric.transforms import NormalizeFeatures    




#Make a directory for the data 
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)


#Put the Cora dataset in a pandas frame
dataset = Planetoid(root= data_dir, name = "Cora")

data = dataset[0]  #This is global storage object. 

#Data Exploration
print("All attributes and methods:", data.__dir__()) #This prints out all the attributes and methods of the dataset

#Let us for example explore the number of nodes and edges in the datset 
print("The number of nodes in the datset: {}".format(data.num_nodes))
print("The number of edges in the datset: {}".format(data.num_edges))   #Because the graph is bi-directional, each edge is counted twice.


#Let's see how edge information is stored. For example, we want to see the edge held by the 30th node.

edge_index = data.edge_index.numpy()
print("The shape of the edge index: {}".format(edge_index.shape))  #This is a [2, data.num_edges] array

#Let us have a look at the edge held by the 30th node 
edge_example = edge_index[:, np.where(edge_index[0]==30)[0]]
print("Edge Example: {}".format(edge_example))


#Now, let us draw a network centered on this node.


#First, I create a directory to store the figures 
figure_dir = "./figures"
os.makedirs(figure_dir, exist_ok=True)

#Here, I store the unique nodes that are connected to the 30th node
node_example = np.unique(edge_example.flatten())


plt.figure(figsize = (10,6))
G = nx.Graph()
G.add_nodes_from(node_example)
G.add_edges_from(list(zip(edge_example[0], edge_example[1])))
nx.draw_networkx(G,  with_labels = True)
plt.savefig(figure_dir + "/example_edge.png")

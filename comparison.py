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
import termtables as tt
import torch.nn.functional as F


monitor = True
plot = True 
loaded_data_LightGCN = torch.load("./outputs/LightGCN.pt")
loaded_data_GCN = torch.load("./outputs/GCN.pt")
loaded_data_SAGE = torch.load("./outputs/GraphSAGE.pt")


if monitor:
    string = tt.to_string(
    [["GCN", loaded_data_GCN["test_precisioin"].item(),  loaded_data_GCN["test_recall"].item()],
     ["LightGCN", loaded_data_LightGCN["test_precisioin"].item(),  loaded_data_LightGCN["test_recall"].item()], 
     ["GraphSAGE", loaded_data_SAGE["test_precisioin"].item(),  loaded_data_SAGE["test_recall"].item()]
     ],
    header=["Method", "Precision", "Recall"],
    style=tt.styles.ascii_thin_double,
    # alignment="ll",
    # padding=(0, 1),
    )
    print(string)
    
if plot: 
    #get the number of epochs 
    num_epochs_LightGCN = loaded_data_LightGCN["num_epoch"]
    LightGCN_train_losses = loaded_data_LightGCN["train_losses"]
    LightGCN_val_losses = loaded_data_LightGCN["val_losses"]
    
    num_epochs_GCN = loaded_data_GCN["num_epoch"]
    GCN_train_losses = loaded_data_GCN["train_losses"]
    GCN_val_losses = loaded_data_GCN["val_losses"]
    
    num_epochs_SAGE = loaded_data_SAGE["num_epoch"]
    SAGE_train_losses = loaded_data_SAGE["train_losses"]
    SAGE_val_losses = loaded_data_SAGE["val_losses"]
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.plot(np.arange(1, num_epochs_LightGCN+1)   ,LightGCN_train_losses, label="LightGCN Train Loss" )
    plt.plot(np.arange(1, num_epochs_LightGCN+1)   ,LightGCN_val_losses, label="LightGCN Validation Loss" )
    plt.grid()
    plt.legend()
    plt.xlabel("Epochs")
    plt.xlabel("BPR Loss")
    plt.savefig("./figures/LightGCN_losses.png")
    plt.show()
    #------------------------------
    plt.figure(figsize=(10, 6), dpi=80)
    plt.plot(np.arange(1, num_epochs_GCN+1)   ,GCN_train_losses, label="GCN Train Loss" )
    plt.plot(np.arange(1, num_epochs_GCN+1)   ,GCN_val_losses, label="GCN Validation Loss" )
    plt.grid()
    plt.legend()
    plt.xlabel("Epochs")
    plt.xlabel("BPR Loss")
    plt.savefig("./figures/GCN_losses.png")
    plt.show()
    
    plt.figure(figsize=(10, 6), dpi=80)
    plt.plot(np.arange(1, num_epochs_SAGE+1)   ,SAGE_train_losses, label="GraphSAGE Train Loss" )
    plt.plot(np.arange(1, num_epochs_SAGE+1)   ,SAGE_val_losses, label="GraphSAGE Validation Loss" )
    plt.grid()
    plt.legend()
    plt.xlabel("Epochs")
    plt.xlabel("BPR Loss")
    plt.savefig("./figures/GCN_losses.png")
    plt.show()
    
    
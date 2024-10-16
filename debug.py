import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# Number of users and items
num_users = 1000
num_items = 1000
num_nodes = num_users + num_items

# Creating input features for nodes
# For simplicity, let's use a single feature for each node
in_channels = 1
x = torch.ones((num_nodes, in_channels))  # Dummy features, replace with actual embeddings if available

# Creating edges (user-item interactions)
# Assume we have the following edges
edge_index = torch.tensor([[0, 1, 2, 0],  # User indices
                            [1, 2, 0, 3]],  # Item indices
                           dtype=torch.long)  # Example edges (user-item pairs)

# Example target ratings corresponding to these edges
targets = torch.tensor([5.0, 3.0, 4.0, 2.0], dtype=torch.float32)  # Ratings for these edges

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)  # Output will be a single value (rating)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x  # Shape: (num_nodes, 1)

# Initialize model
model = GNNModel(in_channels=1, hidden_channels=16)

# Forward pass
out = model(x, edge_index)  # out shape: (num_nodes, 1)

# Get the predictions for the edges
# out[edge_index[0]] gives the output for the source nodes (users)
predictions = out[edge_index[0]]  # Shape will be (num_edges, 1)
print("Output shape for edges:", predictions.shape)  # Should be (num_edges, 1)

# Ensure targets shape matches predictions
print("Targets shape:", targets.shape)  # Should be (num_edges,)

# Verify alignment before loss calculation
assert predictions.shape[0] == targets.shape[0], "Output and target shapes do not match"

# Calculate loss
loss_fn = nn.MSELoss()
loss = loss_fn(predictions.flatten(), targets)  # Calculate loss
print("Loss:", loss.item())


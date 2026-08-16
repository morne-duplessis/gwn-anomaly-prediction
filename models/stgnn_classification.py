import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import List, Dict, Any

class GCNLayer(nn.Module):
   def __init__(self, in_features, out_features):
       super().__init__()
       self.projection = nn.Linear(in_features, out_features)

   def forward(self, x, adj):
       # x shape: [Batch, Nodes, Features]
       # adj shape: [Nodes, Nodes]
       support = torch.einsum('nn,bnf->bnf', adj, x)
       return self.projection(support)

class Chomp1d(nn.Module):
   """Removes the padding on the right to make the convolutions causal."""

   def __init__(self, chomp_size):
       super(Chomp1d, self).__init__()
       self.chomp_size = chomp_size

   def forward(self, x):
       return x[:, :, :-self.chomp_size].contiguous()

class TCNLayer(nn.Module):
   """A multi-layer Temporal Convolutional Network block."""
   def __init__(self, in_channels, num_channels, kernel_size=3, dropout=0.2):
       super(TCNLayer, self).__init__()
       # if not isinstance(num_channels, list):
       #     num_channels = [int(num_channels)]
       layers = []
       num_levels = len(num_channels)


       for i in range(num_levels):

           dilation_size = 2 ** i
           in_c = in_channels if i == 0 else num_channels[i-1]
           out_c = num_channels[i]
           padding = (kernel_size - 1) * dilation_size
           conv = nn.Conv1d(in_c, out_c, kernel_size, stride=1, padding=padding, dilation=dilation_size)
           chomp = Chomp1d(padding)
           relu = nn.ReLU()
           drop = nn.Dropout(dropout)
           layers += [conv, chomp, relu, drop]

       self.network = nn.Sequential(*layers)
       self.out_channels = num_channels[-1]


   def forward(self, x):
       return self.network(x)


class STGNN(nn.Module):




   def __init__(self, args: Dict[str, Any]):

       super(STGNN, self).__init__()
       self.num_nodes = args.get("node_cnt")
       self.hidden_sizes = args.get("hidden_sizes", [64])
       self.horizon = args.get("horizon")
       self.kernel_size = args.get("kernel_size", 3)
       self.dropout = args.get("dropout_rate", 0.2)
       # Determine final dimensionality out of the TCN
       self.out_dim = self.hidden_sizes[-1] if isinstance(self.hidden_sizes, list) else int(self.hidden_sizes)
       # Learnable Graph: We learn the connections (Adjacency)
       self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, 10))

       # Temporal Layer: Our new TCN layer handling the list of hidden sizes
       self.temporal_tcn = TCNLayer(in_channels=1, num_channels=self.hidden_sizes,
                                    kernel_size=self.kernel_size, dropout=self.dropout)

       # Spatial Layer: Our custom GCN taking the output of the TCN
       self.spatial_gcn = GCNLayer(self.out_dim, self.out_dim)

       # Prediction Head
       self.regressor = nn.Linear(self.out_dim * self.num_nodes, self.horizon)


   def get_adj(self):

       """Calculates a normalized adjacency matrix from node embeddings."""

       # Calculate similarity (Cosine/Dot product)
       adj = torch.matmul(self.node_embeddings, self.node_embeddings.t())
       adj = f.softmax(f.relu(adj), dim=-1)
       return adj

   def forward(self, x):

       """
       x: Input tensor, expected shape (B, F, N, T)
       """
       if x.dim() == 4:
           x = x.squeeze(2) # Remove N dimension (N=1) for milling dataframe compatibility

       # Now Input x is in shape: [Batch, Nodes(Features), Window_Size]
       batch_size, num_nodes, window = x.shape
       # Temporal: Treat each node as a channel to extract time features
       # Reshape for TCN: [Batch * Nodes, 1, Window]
       x_temp = x.view(batch_size * num_nodes, 1, window)

       # Pass through the TCN sequence
       x_temp = self.temporal_tcn(x_temp)

       # Pull the 'last' hidden state of the time window
       x_temp = x_temp[:, :, -1] # [Batch * Nodes, Hidden_out_dim]
       # Reshape for GCN: [Batch, Nodes, Hidden_out_dim]
       x_temp = x_temp.view(batch_size, num_nodes, -1)
       # Spatial: Apply GCN using the learned graph
       adj = self.get_adj()
       spatial_features = self.spatial_gcn(x_temp, adj) # Shape: [Batch, Nodes, Hidden_out_dim]
       # FIX: Flatten spatial features to graph-level features
       graph_features = spatial_features.view(batch_size, -1) # Shape: [Batch, Nodes * Hidden_out_dim]

       # Predict system-level output
       predictions = self.regressor(graph_features) # Shape: [Batch, Horizon]
       # Fix output shape: (Batch, 1, 1, Horizon) targeting the single label
       return predictions.view(batch_size, 1, 1, self.horizon)
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

class MLP(nn.Module):
    def __init__(self, args,
                 node_cnt: int = 1,
                 horizon: int = 12,
                 dropout: float = 0.1,
                 out_dim: int = 1,
                 context_len: int = 4,
                 in_dim: int = 1,
                 classify_node_reduce: str = 'mean',
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.node_cnt = node_cnt
        self.context_len = args.get('window_size')
        self.hidden_sizes = args.get('hidden_sizes')
        self.num_layers = len(self.hidden_sizes)
        self.horizon = horizon
        self.out_dim = out_dim
        self.classify_node_reduce = classify_node_reduce
        self.mlp_layers = nn.ModuleList()
        print("context_len:", self.context_len)
        print("hidden_sizes:", self.hidden_sizes)
        prev_dim = self.in_dim * self.node_cnt * self.context_len

        for idx, h in enumerate(self.hidden_sizes):
            self.mlp_layers.append(nn.Linear(prev_dim, h))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.output_layer = nn.Linear(prev_dim, self.node_cnt * self.horizon * self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass for the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, F, N, T)
                              B: batch, F: features, N: nodes, T: time
        
        Returns:
            torch.Tensor: Output tensor of shape (B, 1, N, H) for binary classification
        """
        B, F, N, T = x.shape
        
        # Flatten the input tensor to (B, F * N * T)
        x = x.view(B, -1)

        for layer in self.mlp_layers:
            x = layer(x)
        logits = self.output_layer(x)
        
        # Reshape to (B, N, H, out_dim)
        logits = logits.view(B, self.node_cnt, self.horizon, self.out_dim)

        # Permute to (B, out_dim, N, H) to match other models
        logits = logits.permute(0, 3, 1, 2).contiguous()

        if self.classify_node_reduce == 'mean':
            logits = torch.mean(logits, dim=2, keepdim=True)
        elif self.classify_node_reduce == 'max':
            logits, _ = torch.max(logits, dim=2, keepdim=True)
        
        return logits

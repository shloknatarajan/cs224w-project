import torch
import torch.nn.functional as F
from torch import nn


class EdgeDecoder(nn.Module):
    """Multi-strategy edge decoder for better link prediction"""
    def __init__(self, hidden_dim, dropout=0.5, use_multi_strategy=True):
        super().__init__()
        self.use_multi_strategy = use_multi_strategy

        if use_multi_strategy:
            # Multiple scoring strategies
            # 1. Hadamard product path
            self.hadamard_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

            # 2. Concatenation path
            self.concat_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

            # 3. Bilinear scoring
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

            # Learnable weights for combining strategies
            self.strategy_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            # Simple decoder: just Hadamard product + MLP
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, z, edge):
        src, dst = edge[:, 0], edge[:, 1]
        z_src, z_dst = z[src], z[dst]

        if self.use_multi_strategy:
            # Apply softmax to strategy weights
            weights = F.softmax(self.strategy_weights, dim=0)

            # Strategy 1: Hadamard product
            hadamard = z_src * z_dst
            score1 = self.hadamard_mlp(hadamard).squeeze()

            # Strategy 2: Concatenation
            concat = torch.cat([z_src, z_dst], dim=1)
            score2 = self.concat_mlp(concat).squeeze()

            # Strategy 3: Bilinear
            score3 = self.bilinear(z_src, z_dst).squeeze()

            # Weighted combination
            score = weights[0] * score1 + weights[1] * score2 + weights[2] * score3
        else:
            # Simple Hadamard scoring
            hadamard = z_src * z_dst
            score = self.mlp(hadamard).squeeze()

        return score


class BaseModel(nn.Module):
    """Base class for all GNN models"""
    def __init__(self, hidden_dim, decoder_dropout=0.3, use_multi_strategy=True):
        super().__init__()
        self.decoder = EdgeDecoder(hidden_dim, dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

    def encode(self, edge_index):
        """Encode nodes to embeddings. Should be implemented by subclasses."""
        raise NotImplementedError

    def decode(self, z, edge):
        """Decode edge scores from node embeddings."""
        return self.decoder(z, edge)

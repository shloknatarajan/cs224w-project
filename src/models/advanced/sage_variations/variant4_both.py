"""
Variant 4: Both Improvements
- 3 layers (NEW)
- YES BatchNorm (NEW)
- dropout=0.0

Hypothesis: Depth + BatchNorm together may provide best results
Combines benefits of both variant 2 and variant 3
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from ...base import BaseModel


class SAGEVariant4Both(BaseModel):
    """GraphSAGE with both 3 layers and BatchNorm"""

    def __init__(self, num_nodes, hidden_dim=128, decoder_dropout=0.0, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

        self.hidden_dim = hidden_dim
        self.num_layers = 3
        self.dropout = 0.0
        self.use_batch_norm = True

        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"SAGE-V4-Both (3L, BN✓, drop=0.0) | "
            f"hidden_dim={hidden_dim}, decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # 3 GraphSAGE layers with BatchNorm
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        ])

    def encode(self, edge_index):
        x = self.emb.weight

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)  # Apply BatchNorm before activation
            x = F.relu(x)

        return x

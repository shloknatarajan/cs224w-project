"""
Variant 3: Add Depth
- 3 layers (NEW)
- NO BatchNorm
- dropout=0.0

Hypothesis: More layers = more hops of aggregation, helpful for dense DDI graph
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from ...base import BaseModel


class SAGEVariant3Depth(BaseModel):
    """GraphSAGE with 3 layers for deeper aggregation"""

    def __init__(self, num_nodes, hidden_dim=128, decoder_dropout=0.0, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)

        self.hidden_dim = hidden_dim
        self.num_layers = 3
        self.dropout = 0.0
        self.use_batch_norm = False

        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"SAGE-V3-Depth (3L, no-BN, drop=0.0) | "
            f"hidden_dim={hidden_dim}, decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # 3 GraphSAGE layers
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
        ])

    def encode(self, edge_index):
        x = self.emb.weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return x

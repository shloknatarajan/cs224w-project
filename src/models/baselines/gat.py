import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from ..base import BaseModel


class GAT(BaseModel):
    """Simple GAT baseline model"""

    def __init__(self, num_nodes, hidden_dim, num_layers=2, heads=4, dropout=0.5, decoder_dropout=0.3, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads

        # Model description with hyperparameters
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"Graph Attention Network with multi-head attention and ELU activation | "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, heads={heads}, dropout={dropout}, "
            f"decoder_dropout={decoder_dropout}, decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: average attention heads
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=0.2))
            else:
                # Hidden layers: concatenate attention heads
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, dropout=0.2))

    def encode(self, edge_index):
        x = self.emb.weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

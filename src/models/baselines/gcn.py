import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from ..base import BaseModel


class GCN(BaseModel):
    """Simple GCN baseline model"""

    def __init__(self, num_nodes, hidden_dim, num_layers=2, dropout=0.5, decoder_dropout=0.3, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # Model description with hyperparameters
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"Graph Convolutional Network with spectral convolutions | "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, "
            f"decoder_dropout={decoder_dropout}, decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

    def encode(self, edge_index):
        x = self.emb.weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import TransformerConv
from ..base import BaseModel


class GraphTransformer(BaseModel):
    """Simple Graph Transformer baseline model"""

    def __init__(self, num_nodes, hidden_dim, num_layers=2, heads=4, dropout=0.5, decoder_dropout=0.3, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # Model description with hyperparameters
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"Graph Transformer with multi-head self-attention and position-aware edge encoding | "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, heads={heads}, dropout={dropout}, "
            f"decoder_dropout={decoder_dropout}, decoder={decoder_type}"
        )

        # Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Transformer layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads))

    def encode(self, edge_index):
        x = self.emb.weight

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

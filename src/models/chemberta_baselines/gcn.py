"""
ChemBERTa-based GCN baseline
Uses 768-dim ChemBERTa embeddings projected to hidden_dim
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from ..base import BaseModel


class ChemBERTaGCN(BaseModel):
    """
    GCN baseline with ChemBERTa embedding input features.

    Architecture:
    - Input: ChemBERTa embeddings (768-dim)
    - Projection: Linear layer to map 768 -> hidden_dim
    - GNN: 2-layer GCN
    - Decoder: Simple dot product or multi-strategy

    Key settings:
    - dropout=0 (default - proven to work for dense graphs)
    - decoder_dropout=0 (default)
    """

    def __init__(self, in_channels=768, hidden_dim=128, num_layers=2,
                 dropout=0.0, decoder_dropout=0.0, use_multi_strategy=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout,
                         use_multi_strategy=use_multi_strategy)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        # Model description
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        self.description = (
            f"ChemBERTa-GCN ({num_layers} layers, dropout={dropout}) | "
            f"in={in_channels}, hidden={hidden_dim}, "
            f"decoder={decoder_type}"
        )

        # Projection layer: ChemBERTa embeddings -> hidden_dim
        self.projection = nn.Linear(in_channels, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

    def encode(self, edge_index, x=None):
        """
        Encode nodes to embeddings using ChemBERTa embeddings.

        Args:
            edge_index: Graph edge index
            x: Node features (ChemBERTa embeddings), shape [num_nodes, in_channels]
        """
        if x is None:
            raise ValueError("ChemBERTaGCN requires input features x (ChemBERTa embeddings)")

        # Project ChemBERTa embeddings to hidden dimension
        x = self.projection(x)
        x = F.relu(x)

        # Apply GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

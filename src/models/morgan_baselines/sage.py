"""
Morgan fingerprint-based GraphSAGE baseline
Uses 2048-dim Morgan fingerprints projected to hidden_dim
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from ..base import BaseModel


class MorganGraphSAGE(BaseModel):
    """
    GraphSAGE baseline with Morgan fingerprint input features.

    Architecture:
    - Input: Morgan fingerprints (2048-dim)
    - Projection: Linear layer to map 2048 -> hidden_dim
    - GNN: 2-layer GraphSAGE
    - Decoder: Simple dot product or multi-strategy

    Key settings:
    - dropout=0 (default - proven to work for dense graphs)
    - decoder_dropout=0 (default)
    """

    def __init__(self, in_channels=2048, hidden_dim=128, num_layers=2,
                 dropout=0.0, decoder_dropout=0.0, use_multi_strategy=False,
                 use_batch_norm=False):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout,
                         use_multi_strategy=use_multi_strategy)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        # Model description
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        bn_str = "BN" if use_batch_norm else "no-BN"
        self.description = (
            f"Morgan-GraphSAGE ({num_layers} layers, dropout={dropout}, {bn_str}) | "
            f"in={in_channels}, hidden={hidden_dim}, "
            f"decoder={decoder_type}"
        )

        # Projection layer: Morgan fingerprints -> hidden_dim
        self.projection = nn.Linear(in_channels, hidden_dim)

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index, x=None):
        """
        Encode nodes to embeddings using Morgan fingerprints.

        Args:
            edge_index: Graph edge index
            x: Node features (Morgan fingerprints), shape [num_nodes, in_channels]
        """
        if x is None:
            raise ValueError("MorganGraphSAGE requires input features x (Morgan fingerprints)")

        # Project Morgan fingerprints to hidden dimension
        x = self.projection(x)
        x = F.relu(x)

        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Apply batch normalization if enabled
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            x = F.relu(x)

            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

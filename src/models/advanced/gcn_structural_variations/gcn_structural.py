import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from ...base import BaseModel


class GCNStructural(BaseModel):
    """
    Advanced GCN model with structural features.

    Features:
    - Incorporates structural features (degree, clustering, core number, PageRank, neighbor stats)
    - Batch normalization for stable training
    - Residual connections for better gradient flow
    - Edge dropout for regularization
    - Feature projection for structural features
    """

    def __init__(self, num_nodes, hidden_dim, num_layers=3, dropout=0.4, decoder_dropout=0.3,
                 use_structural_features=True, num_structural_features=6, structural_features=None,
                 use_multi_strategy=True):
        super().__init__(hidden_dim, decoder_dropout=decoder_dropout, use_multi_strategy=use_multi_strategy)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_structural_features = use_structural_features
        self.num_structural_features = num_structural_features
        self.structural_features = structural_features

        # Model description with hyperparameters
        decoder_type = "multi-strategy" if use_multi_strategy else "simple"
        struct_status = f"with {num_structural_features} structural features" if use_structural_features else "without structural features"
        self.description = (
            f"GCN {struct_status}, batch norm, residual connections, and edge dropout | "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, "
            f"decoder_dropout={decoder_dropout}, decoder={decoder_type}"
        )

        # Learnable embeddings with improved initialization
        emb_dim = hidden_dim - num_structural_features if use_structural_features else hidden_dim
        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        # Feature projection layer to transform structural features
        if use_structural_features:
            self.feature_proj = nn.Sequential(
                nn.Linear(num_structural_features, num_structural_features),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )

        # Multiple GCN layers with residual connections
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, edge_index):
        x = self.emb.weight

        # Concatenate structural features with learnable embeddings
        if self.use_structural_features and self.structural_features is not None:
            struct_feats = self.feature_proj(self.structural_features)
            if self.training:
                struct_feats = F.dropout(struct_feats, p=0.1, training=True)
            x = torch.cat([x, struct_feats], dim=1)

        # Moderate edge dropout for regularization - only during training
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection with scaling (for all layers except first)
            if i > 0:
                x = x + 0.5 * x_prev

        return x

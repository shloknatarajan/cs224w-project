"""
SEAL (Subgraph Extraction and Labeling) Model for Link Prediction

Reference:
M. Zhang and Y. Chen. "Link Prediction Based on Graph Neural Networks."
NeurIPS 2018. https://arxiv.org/abs/1802.09691

SEAL learns from local subgraph structures around links rather than
global graph embeddings, making it particularly effective for link prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import SortAggregation


class SEALGCN(nn.Module):
    """
    SEAL model using GCN layers.

    Architecture:
    1. Node embedding layer (learns from DRNL labels)
    2. Multiple GCN layers
    3. Global pooling (SortPool recommended in paper)
    4. MLP classifier
    """
    def __init__(self, max_z=1000, hidden_dim=32, num_layers=3, k=30, dropout=0.5,
                 pooling='sort', use_feature=False, node_feature_dim=0):
        """
        Args:
            max_z: Maximum DRNL label value
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            k: k value for SortPool (number of nodes to keep)
            dropout: Dropout rate
            pooling: Pooling method ('sort', 'add', 'mean', 'max')
            use_feature: Whether to use node features in addition to DRNL labels
            node_feature_dim: Dimension of node features (if use_feature=True)
        """
        super().__init__()

        self.max_z = max_z
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k = k
        self.dropout = dropout
        self.pooling = pooling
        self.use_feature = use_feature

        # Node embedding (DRNL labels -> embeddings)
        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)

        # If using additional node features
        if use_feature:
            self.feature_encoder = nn.Linear(node_feature_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Initialize pooling layer
        if pooling == 'sort':
            self.sort_pool = SortAggregation(k)
            pool_dim = k * hidden_dim
        else:
            pool_dim = hidden_dim

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.description = (
            f"SEAL-GCN (layers={num_layers}, hidden={hidden_dim}, "
            f"pooling={pooling}, k={k if pooling == 'sort' else 'N/A'})"
        )

    def forward(self, data):
        """
        Forward pass on a batch of subgraphs.

        Args:
            data: Batched PyG Data object with:
                - x: DRNL labels (or node features)
                - edge_index: Edge indices
                - batch: Batch assignment vector

        Returns:
            Tensor: Link prediction scores (batch_size,)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embed DRNL labels
        if self.use_feature:
            # x contains both DRNL labels and features
            # Split them (assuming first column is DRNL label)
            z = x[:, 0].long()
            features = x[:, 1:]
            x = self.z_embedding(z) + self.feature_encoder(features)
        else:
            # x only contains DRNL labels
            z = x.long()
            x = self.z_embedding(z)

        # GCN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'sort':
            x = self.sort_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # MLP classifier
        x = self.mlp(x)

        return x.squeeze(-1)


class SEALGIN(nn.Module):
    """
    SEAL model using GIN (Graph Isomorphism Network) layers.

    GIN is theoretically more powerful than GCN for capturing graph structure,
    making it a good choice for SEAL.
    """
    def __init__(self, max_z=1000, hidden_dim=32, num_layers=3, k=30, dropout=0.5,
                 pooling='sort', use_feature=False, node_feature_dim=0):
        """
        Args:
            max_z: Maximum DRNL label value
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            k: k value for SortPool
            dropout: Dropout rate
            pooling: Pooling method ('sort', 'add', 'mean', 'max')
            use_feature: Whether to use node features in addition to DRNL labels
            node_feature_dim: Dimension of node features (if use_feature=True)
        """
        super().__init__()

        self.max_z = max_z
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k = k
        self.dropout = dropout
        self.pooling = pooling
        self.use_feature = use_feature

        # Node embedding (DRNL labels -> embeddings)
        self.z_embedding = nn.Embedding(max_z + 1, hidden_dim)

        # If using additional node features
        if use_feature:
            self.feature_encoder = nn.Linear(node_feature_dim, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Initialize pooling layer
        if pooling == 'sort':
            self.sort_pool = SortAggregation(k)
            pool_dim = k * hidden_dim
        else:
            pool_dim = hidden_dim

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.description = (
            f"SEAL-GIN (layers={num_layers}, hidden={hidden_dim}, "
            f"pooling={pooling}, k={k if pooling == 'sort' else 'N/A'})"
        )

    def forward(self, data):
        """
        Forward pass on a batch of subgraphs.

        Args:
            data: Batched PyG Data object

        Returns:
            Tensor: Link prediction scores (batch_size,)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embed DRNL labels
        if self.use_feature:
            z = x[:, 0].long()
            features = x[:, 1:]
            x = self.z_embedding(z) + self.feature_encoder(features)
        else:
            z = x.long()
            x = self.z_embedding(z)

        # GIN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == 'sort':
            x = self.sort_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # MLP classifier
        x = self.mlp(x)

        return x.squeeze(-1)

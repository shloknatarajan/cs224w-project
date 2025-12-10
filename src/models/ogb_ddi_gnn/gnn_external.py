"""
Extended OGBL-DDI GNN models with external feature support.

These models extend the reference GCN/SAGE architecture to incorporate
external knowledge from:
- Morgan fingerprints (molecular substructure)
- PubChem properties (physicochemical features)
- ChemBERTa embeddings (pre-trained molecular representations)
- Drug-target interactions (biological context)

The external features are combined with learnable embeddings through
a configurable fusion strategy.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling
from torch_geometric.typing import SparseTensor


class FeatureEncoder(nn.Module):
    """Encode external features to hidden dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class GCNExternal(nn.Module):
    """
    GCN encoder with external feature support.

    Combines learnable embeddings with external features through
    concatenation and projection, then applies GCN layers.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        external_dims: Optional[Dict[str, int]] = None,
        fusion: str = "concat",  # "concat" or "add"
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.external_dims = external_dims or {}
        self.fusion = fusion

        # Learnable node embeddings
        self.emb = nn.Embedding(num_nodes, hidden_channels)

        # External feature encoders
        self.feature_encoders = nn.ModuleDict()
        for name, dim in self.external_dims.items():
            self.feature_encoders[name] = FeatureEncoder(
                input_dim=dim,
                hidden_dim=hidden_channels,
                dropout=dropout,
            )

        # Input projection if using concat fusion
        if fusion == "concat" and self.external_dims:
            num_features = 1 + len(self.external_dims)  # emb + external features
            self.input_proj = nn.Linear(hidden_channels * num_features, hidden_channels)
        else:
            self.input_proj = None

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb.weight)
        for conv in self.convs:
            conv.reset_parameters()
        for encoder in self.feature_encoders.values():
            for layer in encoder.encoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if self.input_proj is not None:
            self.input_proj.reset_parameters()

    def forward(
        self,
        adj_t: SparseTensor,
        external_features: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        # Start with learnable embeddings
        x = self.emb.weight

        # Add external features if provided
        if external_features and self.external_dims:
            encoded_features = [x]

            for name in self.external_dims.keys():
                if name in external_features and external_features[name] is not None:
                    feat = self.feature_encoders[name](external_features[name])
                    encoded_features.append(feat)
                else:
                    # Use zeros for missing features
                    encoded_features.append(torch.zeros_like(x))

            if self.fusion == "concat":
                x = torch.cat(encoded_features, dim=1)
                x = self.input_proj(x)
            else:  # add
                x = sum(encoded_features) / len(encoded_features)

        # Apply GCN layers
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x


class SAGEExternal(nn.Module):
    """
    GraphSAGE encoder with external feature support.

    Combines learnable embeddings with external features through
    concatenation and projection, then applies SAGE layers.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        external_dims: Optional[Dict[str, int]] = None,
        fusion: str = "concat",
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.external_dims = external_dims or {}
        self.fusion = fusion

        # Learnable node embeddings
        self.emb = nn.Embedding(num_nodes, hidden_channels)

        # External feature encoders
        self.feature_encoders = nn.ModuleDict()
        for name, dim in self.external_dims.items():
            self.feature_encoders[name] = FeatureEncoder(
                input_dim=dim,
                hidden_dim=hidden_channels,
                dropout=dropout,
            )

        # Input projection if using concat fusion
        if fusion == "concat" and self.external_dims:
            num_features = 1 + len(self.external_dims)
            self.input_proj = nn.Linear(hidden_channels * num_features, hidden_channels)
        else:
            self.input_proj = None

        # SAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb.weight)
        for conv in self.convs:
            conv.reset_parameters()
        for encoder in self.feature_encoders.values():
            for layer in encoder.encoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if self.input_proj is not None:
            self.input_proj.reset_parameters()

    def forward(
        self,
        adj_t: SparseTensor,
        external_features: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        # Start with learnable embeddings
        x = self.emb.weight

        # Add external features if provided
        if external_features and self.external_dims:
            encoded_features = [x]

            for name in self.external_dims.keys():
                if name in external_features and external_features[name] is not None:
                    feat = self.feature_encoders[name](external_features[name])
                    encoded_features.append(feat)
                else:
                    encoded_features.append(torch.zeros_like(x))

            if self.fusion == "concat":
                x = torch.cat(encoded_features, dim=1)
                x = self.input_proj(x)
            else:  # add
                x = sum(encoded_features) / len(encoded_features)

        # Apply SAGE layers
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x


class LinkPredictor(nn.Module):
    """Edge MLP decoder (same as reference implementation)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train_with_external(
    model: nn.Module,
    predictor: LinkPredictor,
    adj_t: SparseTensor,
    split_edge: Dict[str, Dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    external_features: Optional[Dict[str, Tensor]] = None,
) -> float:
    """
    Training epoch with external feature support.

    Args:
        model: Encoder (GCNExternal or SAGEExternal).
        predictor: Edge decoder.
        adj_t: Sparse adjacency (transposed).
        split_edge: Edge splits from OGB dataset.
        optimizer: Optimizer over encoder + decoder.
        batch_size: Mini-batch size for edge sampling.
        external_features: Dict of external feature tensors.
    """
    device = next(model.parameters()).device

    # Build edge_index from training edges for negative sampling
    # Handle both SparseTensor and regular adjacency formats
    if hasattr(adj_t, 'coo'):
        row, col, _ = adj_t.coo()
        edge_index = torch.stack([col, row], dim=0)
    else:
        # Fallback: build from training edges (undirected)
        train_edge = split_edge["train"]["edge"].to(device)
        edge_index = torch.cat([
            train_edge.t(),
            train_edge.t().flip(0)  # Add reverse edges for undirected
        ], dim=1)

    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"].to(device)

    total_loss = 0.0
    total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        h = model(adj_t, external_features)
        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(
            edge_index,
            num_nodes=model.num_nodes,
            num_neg_samples=perm.size(0),
            method="dense",
        )

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test_with_external(
    model: nn.Module,
    predictor: LinkPredictor,
    adj_t: SparseTensor,
    split_edge: Dict[str, Dict[str, Tensor]],
    evaluator,
    batch_size: int,
    external_features: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Evaluation with external feature support."""
    device = next(model.parameters()).device

    model.eval()
    predictor.eval()

    h = model(adj_t, external_features)

    pos_train_edge = split_edge["eval_train"]["edge"].to(device)
    pos_valid_edge = split_edge["valid"]["edge"].to(device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(device)
    pos_test_edge = split_edge["test"]["edge"].to(device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(device)

    def _score_edges(edge_iter: Iterable[int], edges: Tensor) -> Tensor:
        preds = []
        for perm in DataLoader(edge_iter, batch_size):
            edge = edges[perm].t()
            preds.append(predictor(h[edge[0]], h[edge[1]]).squeeze().cpu())
        return torch.cat(preds, dim=0)

    pos_train_pred = _score_edges(range(pos_train_edge.size(0)), pos_train_edge)
    pos_valid_pred = _score_edges(range(pos_valid_edge.size(0)), pos_valid_edge)
    neg_valid_pred = _score_edges(range(neg_valid_edge.size(0)), neg_valid_edge)
    pos_test_pred = _score_edges(range(pos_test_edge.size(0)), pos_test_edge)
    neg_test_pred = _score_edges(range(neg_test_edge.size(0)), neg_test_edge)

    results: Dict[str, Tuple[float, float, float]] = {}
    for k in [10, 20, 30]:
        evaluator.K = k
        train_hits = evaluator.eval(
            {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred}
        )[f"hits@{k}"]
        valid_hits = evaluator.eval(
            {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred}
        )[f"hits@{k}"]
        test_hits = evaluator.eval(
            {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred}
        )[f"hits@{k}"]

        results[f"Hits@{k}"] = (train_hits, valid_hits, test_hits)

    return results

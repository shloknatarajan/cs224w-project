"""
GCNAdvanced: Advanced GCN model for drug-drug interaction prediction.

This is the "Advanced GCN" architecture, featuring:
- 2-layer GCN encoder with cached convolutions
- MLP-based LinkPredictor decoder
- Proper dropout placement (after GCN layers, not during message passing)
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.typing import SparseTensor


class GCNAdvanced(torch.nn.Module):
    """
    Advanced GCN encoder for link prediction.

    Key design choices:
    - Cached GCN convolutions for efficiency on static graphs
    - Dropout applied after activation (not during message passing)
    - Flexible layer depth configuration
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True)
            )
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    """
    MLP-based edge decoder for link prediction.

    Takes node embeddings for a pair of nodes, computes their element-wise
    product, and passes through MLP layers to predict interaction probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

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


def train(
    model: torch.nn.Module,
    predictor: LinkPredictor,
    x: Tensor,
    adj_t: SparseTensor,
    split_edge: Dict[str, Dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
) -> float:
    """
    Single training epoch for GCNAdvanced.

    Args:
        model: GCNAdvanced encoder.
        predictor: LinkPredictor decoder.
        x: Node embeddings (Tensor).
        adj_t: Sparse adjacency (transposed).
        split_edge: Edge splits from OGB dataset.
        optimizer: Optimizer over encoder + decoder + embeddings.
        batch_size: Mini-batch size for edge sampling.
    """
    # Build edge_index from adjacency for negative sampling
    if hasattr(adj_t, 'coo'):
        row, col, _ = adj_t.coo()
        edge_index = torch.stack([col, row], dim=0)
    else:
        # Fallback: build from training edges (undirected)
        device = x.device
        train_edge = split_edge["train"]["edge"].to(device)
        edge_index = torch.cat([
            train_edge.t(),
            train_edge.t().flip(0)  # Add reverse edges for undirected
        ], dim=1)

    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"].to(x.device)

    total_loss = 0.0
    total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)
        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(
            edge_index,
            num_nodes=x.size(0),
            num_neg_samples=perm.size(0),
            method="dense",
        )

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(
    model: torch.nn.Module,
    predictor: LinkPredictor,
    x: Tensor,
    adj_t: SparseTensor,
    split_edge: Dict[str, Dict[str, Tensor]],
    evaluator,
    batch_size: int,
) -> Dict[str, Tuple[float, float, float]]:
    """Evaluation loop for GCNAdvanced."""
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge["eval_train"]["edge"].to(x.device)
    pos_valid_edge = split_edge["valid"]["edge"].to(x.device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(x.device)
    pos_test_edge = split_edge["test"]["edge"].to(x.device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(x.device)

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

"""
Training loop for heuristic-enhanced models (simplified leaderboard approach)
"""
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling, degree

from ..utils.heuristics import compute_edge_heuristics_batched, discretize_heuristics

logger = logging.getLogger(__name__)


@dataclass
class HeuristicRunResult:
    best_val_hits: float
    best_test_hits: float
    best_epoch: int


def train_heuristic_model(
    name: str,
    model: nn.Module,
    data,
    train_pos: torch.Tensor,
    valid_pos: torch.Tensor,
    valid_neg: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    evaluate_fn,
    *,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    eval_every: int = 5,
    patience: int | None = 20,
    batch_size: int = 50000,
    eval_batch_size: int | None = None,
    heuristic_batch_size: int = 10000,
) -> HeuristicRunResult:
    """
    Training loop for heuristic-enhanced models.

    Key differences from standard trainer:
    - Computes graph heuristics for edges
    - Passes heuristics to model decoder
    - Includes degree embeddings
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes

    # Compute node degrees for degree embeddings
    degrees = degree(train_edge_index[0], num_nodes=num_nodes, dtype=torch.long).to(device)

    # Pre-compute heuristics for validation and test sets (they don't change)
    logger.info(f"[{name}] Pre-computing heuristics for validation/test sets...")
    valid_pos_heuristics = compute_edge_heuristics_batched(
        data.edge_index.cpu(), valid_pos.cpu(), num_nodes, batch_size=heuristic_batch_size
    ).to(device)
    valid_neg_heuristics = compute_edge_heuristics_batched(
        data.edge_index.cpu(), valid_neg.cpu(), num_nodes, batch_size=heuristic_batch_size
    ).to(device)
    test_pos_heuristics = compute_edge_heuristics_batched(
        data.edge_index.cpu(), test_pos.cpu(), num_nodes, batch_size=heuristic_batch_size
    ).to(device)
    test_neg_heuristics = compute_edge_heuristics_batched(
        data.edge_index.cpu(), test_neg.cpu(), num_nodes, batch_size=heuristic_batch_size
    ).to(device)

    # Discretize heuristics into bins (LEADERBOARD APPROACH)
    # This converts continuous values to integer bin indices for embedding lookup
    logger.info(f"[{name}] Discretizing heuristics into bins for embedding lookup...")
    valid_pos_heuristics = discretize_heuristics(valid_pos_heuristics)
    valid_neg_heuristics = discretize_heuristics(valid_neg_heuristics)
    test_pos_heuristics = discretize_heuristics(test_pos_heuristics)
    test_neg_heuristics = discretize_heuristics(test_neg_heuristics)

    logger.info(f"[{name}] Heuristics computed. Starting training...")

    best_val = float("-inf")
    best_test = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0

    logger.info(
        f"[{name}] Starting heuristic model training "
        f"(epochs={epochs}, lr={lr}, wd={weight_decay}, batch_size={batch_size})"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Encode nodes (with degree embeddings)
        z = model.encode(train_edge_index, degrees)

        # Sample negative edges
        neg_edges = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_pos.size(0),
            method="sparse",
        ).t().to(device)

        # Compute heuristics for training edges (pos + neg)
        # NOTE: This is expensive! Consider caching or approximations for production
        train_pos_heuristics = compute_edge_heuristics_batched(
            data.edge_index.cpu(), train_pos.cpu(), num_nodes, batch_size=heuristic_batch_size
        ).to(device)
        train_neg_heuristics = compute_edge_heuristics_batched(
            data.edge_index.cpu(), neg_edges.cpu(), num_nodes, batch_size=heuristic_batch_size
        ).to(device)

        # Discretize (LEADERBOARD APPROACH)
        train_pos_heuristics = discretize_heuristics(train_pos_heuristics)
        train_neg_heuristics = discretize_heuristics(train_neg_heuristics)

        # Gradient accumulation: process in batches
        num_batches = (train_pos.size(0) + batch_size - 1) // batch_size
        batch_losses_for_logging = []

        for batch_idx, start in enumerate(range(0, train_pos.size(0), batch_size)):
            end = min(start + batch_size, train_pos.size(0))
            pos_batch = train_pos[start:end]
            neg_batch = neg_edges[start:end]
            pos_h_batch = train_pos_heuristics[start:end]
            neg_h_batch = train_neg_heuristics[start:end]

            # Decode with heuristics
            pos_logits = model.decode(z, pos_batch, pos_h_batch)
            neg_logits = model.decode(z, neg_batch, neg_h_batch)

            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            batch_loss = (pos_loss + neg_loss) / num_batches

            # Backward
            batch_loss.backward(retain_graph=(batch_idx < num_batches - 1))

            batch_losses_for_logging.append((pos_loss + neg_loss).detach().cpu().item())

            # Clear intermediate tensors
            del pos_logits, neg_logits, pos_loss, neg_loss, batch_loss

        # Update parameters
        optimizer.step()

        train_loss = sum(batch_losses_for_logging) / len(batch_losses_for_logging)

        # Clear cache
        del z, neg_edges, train_pos_heuristics, train_neg_heuristics
        torch.cuda.empty_cache()

        # Evaluation
        if epoch == 1 or epoch % eval_every == 0:
            model.eval()
            eval_bs = eval_batch_size or batch_size

            with torch.no_grad():
                z_eval = model.encode(train_edge_index, degrees)

                # Evaluate validation
                val_hits = evaluate_heuristic(
                    model, z_eval, valid_pos, valid_neg,
                    valid_pos_heuristics, valid_neg_heuristics, eval_bs
                )

                # Evaluate test
                test_hits = evaluate_heuristic(
                    model, z_eval, test_pos, test_neg,
                    test_pos_heuristics, test_neg_heuristics, eval_bs
                )

            improved = val_hits > best_val
            if improved:
                best_val = val_hits
                best_test = test_hits
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            logger.info(
                f"[{name}] Epoch {epoch:04d} | loss {train_loss:.4f} | "
                f"val@20 {val_hits:.4f} | test@20 {test_hits:.4f} | "
                f"best {best_val:.4f} (ep {best_epoch})"
            )

            if patience is not None and epochs_no_improve >= patience:
                logger.info(f"[{name}] Early stopping at epoch {epoch}")
                break

    logger.info(
        f"[{name}] Done. Best val@20={best_val:.4f} | test@20={best_test:.4f} "
        f"(epoch {best_epoch})"
    )
    return HeuristicRunResult(best_val, best_test, best_epoch)


def evaluate_heuristic(model, z, pos_edges, neg_edges, pos_heuristics, neg_heuristics, batch_size):
    """
    Evaluate heuristic model using Hits@20 metric.
    """
    model.eval()

    eval_batch_size = min(batch_size, 5000)

    # Score positive edges
    pos_preds = []
    for start in range(0, pos_edges.size(0), eval_batch_size):
        end = min(start + eval_batch_size, pos_edges.size(0))
        batch = pos_edges[start:end]
        h_batch = pos_heuristics[start:end]
        preds = model.decode(z, batch, h_batch)
        pos_preds.append(preds.cpu())
    pos_preds = torch.cat(pos_preds, dim=0)

    # Score negative edges
    neg_preds = []
    for start in range(0, neg_edges.size(0), eval_batch_size):
        end = min(start + neg_edges.size(0), neg_edges.size(0))
        batch = neg_edges[start:end]
        h_batch = neg_heuristics[start:end]
        preds = model.decode(z, batch, h_batch)
        neg_preds.append(preds.cpu())
    neg_preds = torch.cat(neg_preds, dim=0)

    # Compute Hits@20
    hits = 0
    for i in range(pos_preds.size(0)):
        pos_score = pos_preds[i]
        num_higher = (neg_preds >= pos_score).sum().item()
        if num_higher < 20:
            hits += 1

    hits_at_20 = hits / pos_preds.size(0)
    return hits_at_20

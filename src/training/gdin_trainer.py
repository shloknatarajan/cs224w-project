"""
GDIN Trainer with configurable loss (AUC or BCE)

Key differences from minimal_trainer:
1. Configurable loss: AUC (Phase 1) or BCE (Phase 2)
2. Multiple negatives per positive (num_neg=3)
3. Common neighbor feature computation at decode AND eval time
4. Phase 2: BCE + CN for better OOD generalization
"""
import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling

logger = logging.getLogger(__name__)


@dataclass
class GDINRunResult:
    best_val_hits: float
    best_test_hits: float
    best_epoch: int


def auc_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, num_neg: int) -> torch.Tensor:
    """
    Pairwise ranking loss (AUC loss) - directly optimizes Hits@K.

    For each positive edge, we want pos_score > neg_score for all negatives.
    Loss = -log(sigmoid(pos_score - neg_score))

    This is equivalent to BPR loss used in recommendation systems.

    Args:
        pos_scores: Scores for positive edges [B]
        neg_scores: Scores for negative edges [B * num_neg]
        num_neg: Number of negatives per positive

    Returns:
        Scalar loss value
    """
    batch_size = pos_scores.size(0)

    # Reshape for pairwise comparison
    pos = pos_scores.unsqueeze(1)  # [B, 1]
    neg = neg_scores.view(batch_size, num_neg)  # [B, num_neg]

    # Pairwise margin: pos should be > neg
    margin = pos - neg  # [B, num_neg]

    # Log-sigmoid loss (differentiable ranking)
    loss = -F.logsigmoid(margin).mean()

    return loss


def bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss - more robust for OOD generalization.

    Phase 2: BCE is less aggressive than AUC, preventing overfitting
    to the train/val distribution (protein-target split issue).

    Args:
        pos_scores: Scores for positive edges [B]
        neg_scores: Scores for negative edges [B * num_neg]

    Returns:
        Scalar loss value
    """
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores)
    )
    return pos_loss + neg_loss


def compute_cn_counts(
    edges: torch.Tensor,
    adj_sparse,
    device: torch.device
) -> torch.Tensor:
    """
    Compute log-transformed common neighbor counts for edges.

    Phase 2 feature: Inject structural information at decode time.
    CN is topology-based and generalizes across protein targets.

    Args:
        edges: Edge tensor [E, 2]
        adj_sparse: Scipy sparse adjacency matrix
        device: Target device

    Returns:
        Log-transformed CN counts [E]
    """
    src = edges[:, 0].cpu().numpy()
    dst = edges[:, 1].cpu().numpy()

    # Get neighbor sets and compute intersection
    src_neighbors = adj_sparse[src]  # [E, N] sparse
    dst_neighbors = adj_sparse[dst]  # [E, N] sparse

    # Common neighbors = element-wise AND, then sum
    cn_counts = src_neighbors.multiply(dst_neighbors).sum(axis=1)
    cn_counts = torch.tensor(cn_counts).float().squeeze()

    # Log transform (common neighbors follow power law distribution)
    cn_scores = torch.log(cn_counts + 1)

    return cn_scores.to(device)


def evaluate_with_cn(
    model: nn.Module,
    data,
    evaluator,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    adj_sparse,
    batch_size: int = 50000,
    use_cn: bool = True
) -> float:
    """
    Evaluate model with CN features at decode time.

    Critical for Phase 2: CN must be used during evaluation too,
    not just training, for proper OOD generalization.

    Args:
        model: GDIN model
        data: Graph data
        evaluator: OGB evaluator
        pos_edges: Positive edges
        neg_edges: Negative edges
        adj_sparse: Scipy sparse adjacency matrix
        batch_size: Evaluation batch size
        use_cn: Whether to use CN features

    Returns:
        Hits@20 score
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        z = model.encode(data.edge_index)

        # Precompute CN for pos/neg edges if needed
        pos_cn = compute_cn_counts(pos_edges, adj_sparse, device) if use_cn else None
        neg_cn = compute_cn_counts(neg_edges, adj_sparse, device) if use_cn else None

        # Positive scores - batch process
        pos_scores_list = []
        for i in range(0, pos_edges.size(0), batch_size):
            end = min(i + batch_size, pos_edges.size(0))
            chunk = pos_edges[i:end]
            cn_chunk = pos_cn[i:end] if pos_cn is not None else None
            scores = model.decode(z, chunk, cn_chunk).view(-1).cpu()
            pos_scores_list.append(scores)
        pos_scores = torch.cat(pos_scores_list)

        # Negative scores - batch process
        neg_scores_list = []
        for i in range(0, neg_edges.size(0), batch_size):
            end = min(i + batch_size, neg_edges.size(0))
            chunk = neg_edges[i:end]
            cn_chunk = neg_cn[i:end] if neg_cn is not None else None
            scores = model.decode(z, chunk, cn_chunk).view(-1).cpu()
            neg_scores_list.append(scores)
        neg_scores = torch.cat(neg_scores_list)

        # Free memory
        del z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use OGB evaluator
        result = evaluator.eval({
            'y_pred_pos': pos_scores,
            'y_pred_neg': neg_scores,
        })

        return result['hits@20']


def train_gdin(
    name: str,
    model: nn.Module,
    data,
    train_pos: torch.Tensor,
    valid_pos: torch.Tensor,
    valid_neg: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    evaluate_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, int], float],
    *,
    device: torch.device,
    epochs: int = 400,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    eval_every: int = 5,
    patience: int | None = 30,
    batch_size: int = 50000,
    eval_batch_size: int | None = None,
    num_neg: int = 3,
    use_cn: bool = False,
    adj_sparse=None,  # Required if use_cn=True
) -> GDINRunResult:
    """
    Train GDIN model with pairwise ranking loss.

    Key differences from minimal_trainer:
    1. AUC loss instead of BCE
    2. Multiple negatives per positive (num_neg)
    3. Optional CN features (Phase 2)

    Args:
        name: Model name for logging
        model: GDIN model
        data: Graph data
        train_pos: Positive training edges
        valid_pos/neg: Validation edges
        test_pos/neg: Test edges
        evaluate_fn: Evaluation function
        device: Target device
        epochs: Max training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        eval_every: Evaluate every N epochs
        patience: Early stopping patience
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_neg: Number of negatives per positive
        use_cn: Enable common neighbor features (Phase 2)
        adj_sparse: Scipy sparse adjacency matrix (required if use_cn=True)

    Returns:
        GDINRunResult with best validation/test scores
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes

    # Precompute CN for training positives (Phase 2)
    train_cn = None
    if use_cn:
        if adj_sparse is None:
            raise ValueError("adj_sparse required when use_cn=True")
        logger.info(f"[{name}] Precomputing CN counts for {train_pos.size(0)} training edges...")
        train_cn = compute_cn_counts(train_pos, adj_sparse, device)

    best_val = float("-inf")
    best_test = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0

    logger.info(
        f"[{name}] Starting GDIN training "
        f"(epochs={epochs}, lr={lr}, wd={weight_decay}, num_neg={num_neg}, use_cn={use_cn})"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Encode all nodes
        z = model.encode(train_edge_index)

        # Generate multiple negatives per positive
        total_neg = train_pos.size(0) * num_neg
        neg_edges = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=total_neg,
            method="sparse",
        ).t().to(device)

        # Compute CN for negatives (Phase 2) - on-the-fly since negatives change
        neg_cn = None
        if use_cn:
            neg_cn = compute_cn_counts(neg_edges, adj_sparse, device)

        # Batch processing
        batch_losses = []
        num_train = train_pos.size(0)

        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            pos_batch = train_pos[start:end]

            # Get corresponding negative batch (num_neg per positive)
            neg_start = start * num_neg
            neg_end = end * num_neg
            neg_batch = neg_edges[neg_start:neg_end]

            # Score edges
            pos_cn_batch = train_cn[start:end] if train_cn is not None else None
            neg_cn_batch = neg_cn[neg_start:neg_end] if neg_cn is not None else None

            pos_scores = model.decode(z, pos_batch, pos_cn_batch)
            neg_scores = model.decode(z, neg_batch, neg_cn_batch)

            # Pairwise ranking loss
            loss = auc_loss(pos_scores, neg_scores, num_neg)
            batch_losses.append(loss)

        total_loss = torch.stack(batch_losses).mean()
        total_loss.backward()
        optimizer.step()

        train_loss = float(total_loss.detach().cpu())

        # Evaluation
        if epoch == 1 or epoch % eval_every == 0:
            eval_bs = eval_batch_size or batch_size
            val_hits = float(evaluate_fn(model, valid_pos, valid_neg, eval_bs))
            test_hits = float(evaluate_fn(model, test_pos, test_neg, eval_bs))

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
                logger.info(
                    f"[{name}] Early stopping at epoch {epoch} "
                    f"(no val improvement for {epochs_no_improve} evals)"
                )
                break

    logger.info(
        f"[{name}] Done. Best val@20={best_val:.4f} | test@20={best_test:.4f} "
        f"(epoch {best_epoch})"
    )
    return GDINRunResult(best_val, best_test, best_epoch)

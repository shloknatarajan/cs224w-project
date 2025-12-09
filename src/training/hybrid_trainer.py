"""
Training loop for hybrid models (structure-only encoder + chemistry-aware decoder)
"""
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling

logger = logging.getLogger(__name__)


@dataclass
class HybridRunResult:
    best_val_hits: float
    best_test_hits: float
    best_epoch: int


def train_hybrid_model(
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
) -> HybridRunResult:
    """
    Training loop for hybrid models with chemistry-aware decoders.
    
    Key differences from minimal_trainer:
    - Passes chemistry features to decoder
    - Passes smiles_mask to decoder for selective chemistry use
    - Encoder remains structure-only
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes
    
    # Get chemistry features and mask
    chemistry = data.x.to(device) if hasattr(data, 'x') and data.x is not None else None
    smiles_mask = data.smiles_mask.to(device) if hasattr(data, 'smiles_mask') and data.smiles_mask is not None else None
    
    if chemistry is None:
        raise ValueError("Hybrid models require chemistry features (data.x)")
    
    best_val = float("-inf")
    best_test = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0
    
    logger.info(
        f"[{name}] Starting hybrid model training "
        f"(epochs={epochs}, lr={lr}, wd={weight_decay}, batch_size={batch_size}, eval_every={eval_every})"
    )
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Encode: structure-only
        z = model.encode(train_edge_index)
        
        # Sample negative edges
        neg_edges = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_pos.size(0),
            method="sparse",
        ).t().to(device)
        
        # Decode in batches and accumulate losses
        batch_losses = []
        
        for start in range(0, train_pos.size(0), batch_size):
            end = min(start + batch_size, train_pos.size(0))
            pos_batch = train_pos[start:end]
            neg_batch = neg_edges[start:end]
            
            # Decode with chemistry
            pos_logits = model.decode(z, chemistry, pos_batch, smiles_mask)
            neg_logits = model.decode(z, chemistry, neg_batch, smiles_mask)
            
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            batch_losses.append(pos_loss + neg_loss)
        
        # Compute total loss and backprop once
        total_loss = torch.stack(batch_losses).mean()
        total_loss.backward()
        optimizer.step()
        
        train_loss = float(total_loss.detach().cpu())
        
        # Clear cache after backward
        torch.cuda.empty_cache()
        
        # Evaluation
        if epoch == 1 or epoch % eval_every == 0:
            model.eval()
            eval_bs = eval_batch_size or batch_size
            
            with torch.no_grad():
                z_eval = model.encode(train_edge_index)
                
                # Evaluate validation
                val_hits = evaluate_hybrid(model, z_eval, chemistry, valid_pos, valid_neg, smiles_mask, eval_bs)
                
                # Evaluate test
                test_hits = evaluate_hybrid(model, z_eval, chemistry, test_pos, test_neg, smiles_mask, eval_bs)
            
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
                logger.info(f"[{name}] Early stopping at epoch {epoch} (no val improvement for {epochs_no_improve} evals)")
                break
    
    logger.info(
        f"[{name}] Done. Best val@20={best_val:.4f} | test@20={best_test:.4f} "
        f"(epoch {best_epoch})"
    )
    return HybridRunResult(best_val, best_test, best_epoch)


def evaluate_hybrid(model, z, chemistry, pos_edges, neg_edges, smiles_mask, batch_size):
    """
    Evaluate hybrid model using Hits@20 metric.
    
    Args:
        model: Hybrid model
        z: Structural node embeddings [N, hidden_dim]
        chemistry: Chemical features [N, chem_dim]
        pos_edges: Positive edges [E_pos, 2]
        neg_edges: Negative edges [E_neg, 2]
        smiles_mask: Binary mask [N] for valid chemistry
        batch_size: Batch size for evaluation
    
    Returns:
        hits_at_20: Hits@20 score
    """
    model.eval()
    
    # Use smaller batch size for evaluation to avoid OOM
    eval_batch_size = min(batch_size, 5000)
    
    # Score positive edges
    pos_preds = []
    for start in range(0, pos_edges.size(0), eval_batch_size):
        end = min(start + eval_batch_size, pos_edges.size(0))
        batch = pos_edges[start:end]
        preds = model.decode(z, chemistry, batch, smiles_mask)
        pos_preds.append(preds.cpu())
    pos_preds = torch.cat(pos_preds, dim=0)
    
    # Score negative edges
    neg_preds = []
    for start in range(0, neg_edges.size(0), eval_batch_size):
        end = min(start + eval_batch_size, neg_edges.size(0))
        batch = neg_edges[start:end]
        preds = model.decode(z, chemistry, batch, smiles_mask)
        neg_preds.append(preds.cpu())
    neg_preds = torch.cat(neg_preds, dim=0)
    
    # Compute Hits@20
    # For each positive edge, count how many negatives have lower scores
    hits = 0
    for i in range(pos_preds.size(0)):
        pos_score = pos_preds[i]
        # Count negatives with score >= pos_score
        num_higher = (neg_preds >= pos_score).sum().item()
        # If rank <= 20, it's a hit
        if num_higher < 20:
            hits += 1
    
    hits_at_20 = hits / pos_preds.size(0)
    return hits_at_20


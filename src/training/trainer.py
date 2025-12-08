import torch
import torch.nn.functional as F
import logging
from torch_geometric.utils import negative_sampling
from .losses import hard_negative_mining, k_hop_negative_sampling

logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters.
    Provides smoother predictions and better generalization.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters after each training step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_model(name, model, data, train_pos, valid_pos, valid_neg, test_pos, test_neg,
                num_nodes, evaluate_fn, device='cpu', epochs=200, lr=0.01, patience=20,
                eval_every=5, use_hard_negatives=True, hard_neg_ratio=0.3, batch_size=20000,
                eval_batch_size=50000, gradient_accumulation_steps=3, weight_decay=5e-5,
                neg_sampling_strategy='hard', k_hop=2):
    """
    Train model with early stopping, validation, and hard negative mining.

    Args:
        name: Model name for logging
        model: Model to train
        data: Graph data object
        train_pos: Training positive edges
        valid_pos: Validation positive edges
        valid_neg: Validation negative edges
        test_pos: Test positive edges
        test_neg: Test negative edges
        num_nodes: Number of nodes in the graph
        evaluate_fn: Evaluation function
        device: Device to train on
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        eval_every: Evaluate every N epochs
        use_hard_negatives: Whether to use hard negative mining
        hard_neg_ratio: Ratio of hard negatives to use
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        gradient_accumulation_steps: Number of steps to accumulate gradients
        weight_decay: Weight decay for optimizer
        neg_sampling_strategy: Negative sampling strategy ('random', 'hard', 'khop', 'mixed')
        k_hop: Number of hops for k-hop negative sampling

    Returns:
        tuple: (best_val_hits, best_test_hits)
    """
    logger.info(f"Starting training for {name} (epochs={epochs}, lr={lr}, patience={patience}, hard_neg={use_hard_negatives})")
    if use_hard_negatives:
        logger.info(f"Negative sampling: strategy={neg_sampling_strategy}, ratio={hard_neg_ratio}, k_hop={k_hop if neg_sampling_strategy in ['khop', 'mixed'] else 'N/A'}")
    if hasattr(model, 'description'):
        logger.info(f"Model description: {model.description}")
    logger.info(f"Memory optimization: batch_size={batch_size}, gradient_accumulation={gradient_accumulation_steps}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=True
    )

    # Initialize EMA
    ema = ExponentialMovingAverage(model, decay=0.999)
    logger.info("Initialized EMA with decay=0.999 for stable checkpointing")

    best_val_hits = 0
    best_test_hits = 0
    best_epoch = 0
    epochs_no_improve = 0

    warmup_epochs = 50

    for epoch in range(1, epochs + 1):
        model.train()

        # Gradient accumulation
        total_loss = 0.0
        num_samples_per_step = train_pos.size(0) // gradient_accumulation_steps

        for accum_step in range(gradient_accumulation_steps):
            # Get micro-batch
            start_idx = accum_step * num_samples_per_step
            end_idx = start_idx + num_samples_per_step if accum_step < gradient_accumulation_steps - 1 else train_pos.size(0)
            pos_batch = train_pos[start_idx:end_idx]

            # Encode graph
            z = model.encode(data.edge_index)

            # Generate negatives based on strategy
            num_negatives = pos_batch.size(0)
            if use_hard_negatives and epoch > warmup_epochs:
                num_hard = int(num_negatives * hard_neg_ratio)
                num_random = num_negatives - num_hard

                # Select hard negative strategy
                if neg_sampling_strategy == 'hard':
                    # Original hard negative mining
                    hard_neg = hard_negative_mining(model, z, data.edge_index, num_nodes, num_hard, top_k_ratio=0.3, device=device)
                elif neg_sampling_strategy == 'khop':
                    # K-hop negative sampling
                    hard_neg = k_hop_negative_sampling(data.edge_index, num_nodes, num_hard, k=k_hop, device=device)
                elif neg_sampling_strategy == 'mixed':
                    # Mix of score-based hard negatives and k-hop negatives
                    num_score_hard = num_hard // 2
                    num_khop_hard = num_hard - num_score_hard
                    score_hard = hard_negative_mining(model, z, data.edge_index, num_nodes, num_score_hard, top_k_ratio=0.3, device=device)
                    khop_hard = k_hop_negative_sampling(data.edge_index, num_nodes, num_khop_hard, k=k_hop, device=device)
                    hard_neg = torch.cat([score_hard, khop_hard], dim=0)
                    del score_hard, khop_hard
                else:
                    # Default to random
                    hard_neg = negative_sampling(
                        edge_index=data.edge_index,
                        num_nodes=num_nodes,
                        num_neg_samples=num_hard
                    ).t().to(device)

                # Random negatives
                random_neg = negative_sampling(
                    edge_index=data.edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=num_random
                ).t().to(device)

                neg_samples = torch.cat([hard_neg, random_neg], dim=0)
                del hard_neg, random_neg
            else:
                neg_samples = negative_sampling(
                    edge_index=data.edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=num_negatives
                ).t().to(device)

            # Memory-efficient batch decoding
            pos_out_list = []
            for i in range(0, pos_batch.size(0), batch_size):
                chunk = pos_batch[i:i+batch_size]
                scores = model.decode(z, chunk)
                pos_out_list.append(scores)
                del scores
            pos_out = torch.cat(pos_out_list)
            del pos_out_list

            neg_out_list = []
            for i in range(0, neg_samples.size(0), batch_size):
                chunk = neg_samples[i:i+batch_size]
                scores = model.decode(z, chunk)
                neg_out_list.append(scores)
                del scores
            neg_out = torch.cat(neg_out_list)
            del neg_out_list

            # Loss computation
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_out, torch.ones_like(pos_out)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_out, torch.zeros_like(neg_out)
            )
            loss = pos_loss + neg_loss

            # Add L2 regularization on embeddings
            emb_reg_weight = 0.005
            emb_reg_loss = emb_reg_weight * torch.norm(model.emb.weight, p=2)
            loss = loss + emb_reg_loss

            # Scale loss
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Clear intermediate tensors
            del z, neg_samples, pos_out, neg_out, pos_loss, neg_loss, emb_reg_loss, loss
            # Clear cache for CUDA devices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Update EMA
        ema.update()

        # Evaluation
        if epoch % eval_every == 0 or epoch == 1:
            ema.apply_shadow()
            val_hits = evaluate_fn(model, valid_pos, valid_neg, batch_size=eval_batch_size)
            test_hits = evaluate_fn(model, test_pos, test_neg, batch_size=eval_batch_size)
            ema.restore()

            # Clear cache for CUDA and MPS devices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            scheduler.step(val_hits)

            improved = val_hits > best_val_hits
            if improved:
                best_val_hits = val_hits
                best_test_hits = test_hits
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            current_lr = optimizer.param_groups[0]['lr']
            improvement_marker = "🔥" if improved else ""
            hard_neg_status = f"[Hard Neg]" if use_hard_negatives and epoch > warmup_epochs else "[Random Neg]"
            logger.info(
                f"{name} Epoch {epoch:04d}/{epochs} {hard_neg_status} | "
                f"Loss: {total_loss:.4f} | "
                f"Val Hits@20: {val_hits:.4f} | "
                f"Test Hits@20: {test_hits:.4f} | "
                f"Best Val: {best_val_hits:.4f} (epoch {best_epoch}) | "
                f"LR: {current_lr:.6f} {improvement_marker}"
            )

            if epochs_no_improve >= patience:
                logger.info(f"{name}: Early stopping at epoch {epoch} (no improvement for {patience} eval steps)")
                break

    logger.info(f"{name} FINAL: Best Val Hits@20 = {best_val_hits:.4f} | Test Hits@20 = {best_test_hits:.4f} (at epoch {best_epoch})")
    return best_val_hits, best_test_hits

"""
Train SEAL (Subgraph Extraction and Labeling) model for link prediction

SEAL is different from baseline GNN models:
- It extracts local subgraphs around each link
- Uses DRNL node labeling to encode structural roles
- Trains a GNN classifier on these subgraphs
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import os
from datetime import datetime

from src.data import load_dataset, SEALDataset, seal_collate_fn
from src.models.advanced import SEALGCN, SEALGIN
from torch_geometric.utils import negative_sampling

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'logs/seal_{timestamp}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{log_dir}/seal.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
logger.info(f"Using device: {device}")
logger.info(f"Logging results to: {log_filename}")

# Load dataset
data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device='cpu')

# Extract edges
train_pos = split_edge['train']['edge']
valid_pos = split_edge['valid']['edge']
valid_neg = split_edge['valid']['edge_neg']
test_pos = split_edge['test']['edge']
test_neg = split_edge['test']['edge_neg']

# Subsample training data for faster experimentation
SUBSAMPLE_RATIO = 0.1  # Use 10% of training data
num_train = int(len(train_pos) * SUBSAMPLE_RATIO)
train_pos = train_pos[:num_train]
logger.info(f"Subsampling training data to {SUBSAMPLE_RATIO*100:.0f}% ({num_train} edges)")

logger.info(f"Train pos edges: {train_pos.size(0)}")
logger.info(f"Valid pos edges: {valid_pos.size(0)}, Valid neg edges: {valid_neg.size(0)}")
logger.info(f"Test pos edges: {test_pos.size(0)}, Test neg edges: {test_neg.size(0)}")

# Construct training graph (only training edges)
train_edge_index = train_pos.t().contiguous()

# SEAL Hyperparameters
NUM_HOPS = 1  # Number of hops for subgraph extraction
MAX_Z = 1000  # Maximum DRNL label
HIDDEN_DIM = 32  # Hidden dimension
NUM_LAYERS = 3  # Number of GNN layers
K = 30  # SortPool k value
DROPOUT = 0.5
POOLING = 'sort'  # Pooling method: 'sort', 'add', 'mean', 'max'
BATCH_SIZE = 32  # Batch size for subgraphs
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MODEL_TYPE = 'GIN'  # 'GCN' or 'GIN'

logger.info("\n" + "=" * 80)
logger.info("SEAL Configuration")
logger.info("=" * 80)
logger.info(f"Model type: {MODEL_TYPE}")
logger.info(f"Subgraph hops: {NUM_HOPS}")
logger.info(f"Hidden dim: {HIDDEN_DIM}")
logger.info(f"Num layers: {NUM_LAYERS}")
logger.info(f"Pooling: {POOLING} (k={K if POOLING == 'sort' else 'N/A'})")
logger.info(f"Dropout: {DROPOUT}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Learning rate: {LEARNING_RATE}")
logger.info("=" * 80)

# Sample negative edges for training (SEAL needs to balance pos/neg)
# For efficiency, we'll use a subset of possible negative edges
logger.info("Generating training negative samples...")
num_train_neg = train_pos.size(0)  # Same number as positive edges
train_neg = negative_sampling(
    edge_index=train_edge_index,
    num_nodes=num_nodes,
    num_neg_samples=num_train_neg
).t()
logger.info(f"Generated {train_neg.size(0)} training negative samples")

# Create datasets
logger.info("Creating SEAL datasets (this may take a while)...")
train_dataset = SEALDataset(
    train_edge_index, train_pos, train_neg, num_nodes,
    num_hops=NUM_HOPS, max_nodes_per_hop=None
)
logger.info(f"Train dataset created: {len(train_dataset)} samples")

# For validation/test, we'll sample a subset for efficiency
# (Full evaluation would be too slow for SEAL)
MAX_EVAL_SAMPLES = 5000
valid_pos_sample = valid_pos[:MAX_EVAL_SAMPLES]
valid_neg_sample = valid_neg[:MAX_EVAL_SAMPLES]
test_pos_sample = test_pos[:MAX_EVAL_SAMPLES]
test_neg_sample = test_neg[:MAX_EVAL_SAMPLES]

valid_dataset = SEALDataset(
    train_edge_index, valid_pos_sample, valid_neg_sample, num_nodes,
    num_hops=NUM_HOPS, max_nodes_per_hop=None
)
logger.info(f"Valid dataset created: {len(valid_dataset)} samples")

test_dataset = SEALDataset(
    train_edge_index, test_pos_sample, test_neg_sample, num_nodes,
    num_hops=NUM_HOPS, max_nodes_per_hop=None
)
logger.info(f"Test dataset created: {len(test_dataset)} samples")

# Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=seal_collate_fn, num_workers=0  # Set to 0 for compatibility
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=seal_collate_fn, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=seal_collate_fn, num_workers=0
)

# Create model
if MODEL_TYPE == 'GCN':
    model = SEALGCN(
        max_z=MAX_Z, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        k=K, dropout=DROPOUT, pooling=POOLING
    )
elif MODEL_TYPE == 'GIN':
    model = SEALGIN(
        max_z=MAX_Z, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        k=K, dropout=DROPOUT, pooling=POOLING
    )
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

model = model.to(device)
logger.info(f"Model: {model.description}")
logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Verify GPU usage
if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total")
    logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
    logger.info(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def evaluate_seal(model, loader, device):
    """Evaluate SEAL model on a dataset"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = (out > 0).float()
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)

            loss = F.binary_cross_entropy_with_logits(out, batch.y)
            total_loss += loss.item() * batch.y.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


# Training loop
logger.info("\n" + "=" * 80)
logger.info("Starting SEAL Training")
logger.info("=" * 80)

best_val_acc = 0
best_test_acc = 0
best_epoch = 0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(out, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        # Progress tracking - log every 1000 batches
        if num_batches % 1000 == 0:
            current_avg_loss = total_loss / num_batches
            gpu_mem = f" - GPU Mem: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB" if device.type == 'cuda' else ""
            logger.info(f"  Epoch {epoch} - Batch {num_batches}/{len(train_loader)} ({100*num_batches/len(train_loader):.1f}%) - Avg Loss: {current_avg_loss:.4f}{gpu_mem}")

    avg_loss = total_loss / num_batches

    # Evaluation
    if epoch % 1 == 0:  # Evaluate every epoch
        val_acc, val_loss = evaluate_seal(model, valid_loader, device)
        test_acc, test_loss = evaluate_seal(model, test_loader, device)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        improvement_marker = "🔥" if improved else ""
        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} (Loss: {val_loss:.4f}) | "
            f"Test Acc: {test_acc:.4f} (Loss: {test_loss:.4f}) | "
            f"Best Val: {best_val_acc:.4f} (epoch {best_epoch}) {improvement_marker}"
        )

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

# Final results
logger.info("\n" + "=" * 80)
logger.info("SEAL FINAL RESULTS")
logger.info("=" * 80)
logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
logger.info(f"Test Accuracy (at best val): {best_test_acc:.4f}")
logger.info(f"Best epoch: {best_epoch}")
logger.info("=" * 80)
logger.info(f"Results logged to: {log_filename}")
logger.info("\nNote: SEAL metrics are accuracy on sampled edges, not Hits@K like baselines")

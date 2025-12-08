"""
Train advanced GCN model with structural features
"""
import torch
import logging
import os
from datetime import datetime

from src.data import load_dataset, compute_structural_features
from src.models.advanced import GCNStructuralV4
from src.training import train_model
from src.evals import evaluate

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'logs/advanced_{timestamp}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{log_dir}/advanced.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device setup - support CUDA (Linux), MPS (macOS), and CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
logger.info(f"Using device: {device}")
logger.info(f"Logging results to: {log_filename}")

# Load dataset
data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device=device)

train_pos = split_edge['train']['edge'].to(device)
valid_pos = split_edge['valid']['edge'].to(device)
valid_neg = split_edge['valid']['edge_neg'].to(device)
test_pos = split_edge['test']['edge'].to(device)
test_neg = split_edge['test']['edge_neg'].to(device)

# Compute structural features
train_edge_index = train_pos.t().contiguous().to(device)
structural_features = compute_structural_features(train_edge_index, num_nodes, device=device)

# Hyperparameters - V4 with multi-strategy decoder
HIDDEN_DIM = 256  # Kept at 256 for capacity
NUM_LAYERS = 3  # 3 layers for good depth
DROPOUT = 0.2  # Kept low for gradient flow
DECODER_DROPOUT = 0.3  # Kept at 0.3
EPOCHS = 400
PATIENCE = 20  # Kept at 20 for efficiency
LEARNING_RATE = 0.005  # Kept at 0.005
WEIGHT_DECAY = 5e-5
USE_HARD_NEGATIVES = True
HARD_NEG_RATIO = 0.05
BATCH_SIZE = 20000
EVAL_BATCH_SIZE = 50000
GRADIENT_ACCUMULATION_STEPS = 3
NUM_STRUCTURAL_FEATURES = 6
USE_MULTI_STRATEGY = True  # ENABLED - key change from V3
DIVERSITY_WEIGHT = 0.02  # Reduced from 0.05 (we achieved good diversity)

config = {
    'model_name': 'GCN-Structural-V4',
    'hidden_dim': HIDDEN_DIM,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'decoder_dropout': DECODER_DROPOUT,
    'epochs': EPOCHS,
    'patience': PATIENCE,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'use_hard_negatives': USE_HARD_NEGATIVES,
    'hard_neg_ratio': HARD_NEG_RATIO,
    'batch_size': BATCH_SIZE,
    'eval_batch_size': EVAL_BATCH_SIZE,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    'num_structural_features': NUM_STRUCTURAL_FEATURES,
    'use_multi_strategy': USE_MULTI_STRATEGY,
    'diversity_weight': DIVERSITY_WEIGHT,
    'num_nodes': num_nodes,
    'train_edges': train_pos.size(0),
    'val_edges': valid_pos.size(0),
    'test_edges': test_pos.size(0)
}

logger.info("=" * 80)
logger.info(f"MODEL CONFIGURATION - {config['model_name']}")
logger.info("=" * 80)
logger.info("Architecture:")
logger.info(f"  Model: {config['model_name']}")
logger.info(f"  Hidden Dim: {config['hidden_dim']}, Layers: {config['num_layers']}")
logger.info(f"  Dropout: {config['dropout']}, Decoder Dropout: {config['decoder_dropout']}")
logger.info(f"  Structural Features: {config['num_structural_features']} dims")
logger.info(f"  Multi-strategy decoder: {config['use_multi_strategy']}")
logger.info(f"  Diversity Loss Weight: {config['diversity_weight']}")
logger.info("")
logger.info("Training:")
logger.info(f"  Epochs: {config['epochs']}, Patience: {config['patience']}")
logger.info(f"  Learning Rate: {config['learning_rate']}, Weight Decay: {config['weight_decay']}")
logger.info(f"  Hard Negatives: {config['use_hard_negatives']} (ratio={config['hard_neg_ratio']})")
logger.info(f"  Batch Size: {config['batch_size']}, Eval Batch: {config['eval_batch_size']}")
logger.info(f"  Gradient Accumulation Steps: {config['gradient_accumulation_steps']}")
logger.info("")
logger.info("Dataset:")
logger.info(f"  Nodes: {config['num_nodes']}")
logger.info(f"  Train Edges: {config['train_edges']}")
logger.info(f"  Val Edges: {config['val_edges']}, Test Edges: {config['test_edges']}")
logger.info("=" * 80)

# Create evaluation function wrapper
def evaluate_fn(model, pos_edges, neg_edges, batch_size):
    return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

# Train model
logger.info("\n" + "=" * 80)
logger.info("Training Advanced GCN with Structural Features")
logger.info("=" * 80)

model = GCNStructuralV4(
    num_nodes,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT,
    use_structural_features=True,
    num_structural_features=NUM_STRUCTURAL_FEATURES,
    structural_features=structural_features,
    use_multi_strategy=USE_MULTI_STRATEGY,
    diversity_weight=DIVERSITY_WEIGHT
)

gcn_val, gcn_test = train_model(
    config['model_name'],
    model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    num_nodes,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    use_hard_negatives=USE_HARD_NEGATIVES,
    hard_neg_ratio=HARD_NEG_RATIO,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    weight_decay=WEIGHT_DECAY
)

# Final results
logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS")
logger.info("=" * 80)
logger.info(f"Model: {config['model_name']}")
logger.info(f"Configuration: Hidden Dim={config['hidden_dim']}, Layers={config['num_layers']}, Dropout={config['dropout']}")
logger.info("")
logger.info(f"{config['model_name']}:")
logger.info(f"  Validation Hits@20: {gcn_val:.4f}")
logger.info(f"  Test Hits@20: {gcn_test:.4f}")
val_test_gap = gcn_val - gcn_test
logger.info(f"  Val-Test Gap: {val_test_gap:.4f} ({val_test_gap/gcn_val*100:.1f}% relative)")
logger.info("=" * 80)
logger.info(f"Results logged to: {log_filename}")

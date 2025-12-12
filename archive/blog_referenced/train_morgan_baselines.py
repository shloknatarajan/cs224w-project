"""
Train Morgan fingerprint-based baseline models (GCN, GraphSAGE, GraphTransformer, GAT)

This script uses Morgan fingerprints (2048-dim) as input features instead of learnable embeddings.
"""
import torch
import logging
import os
from datetime import datetime

from src.data import load_dataset
from src.models.morgan_baselines import MorganGCN, MorganGraphSAGE, MorganGraphTransformer, MorganGAT
from src.training import train_minimal_baseline
from src.evals import evaluate

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'logs/morgan_baselines_{timestamp}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{log_dir}/morgan_baselines.log'

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

# Load dataset with Morgan fingerprints
# TODO: Update these paths to your actual SMILES CSV and cache file
SMILES_CSV_PATH = "data/smiles.csv"  # CSV with columns: ogb_id, smiles
FEATURE_CACHE_PATH = "data/morgan_features_2048.pt"  # Cache file for faster loading

logger.info("Loading dataset with Morgan fingerprints...")
data, split_edge, num_nodes, evaluator = load_dataset(
    'ogbl-ddi',
    device=device,
    smiles_csv_path=SMILES_CSV_PATH,
    feature_cache_path=FEATURE_CACHE_PATH,
    fp_n_bits=2048,
    fp_radius=2
)

logger.info(f"Dataset loaded with Morgan features: data.x.shape = {data.x.shape}")

train_pos = split_edge['train']['edge'].to(device)
valid_pos = split_edge['valid']['edge'].to(device)
valid_neg = split_edge['valid']['edge_neg'].to(device)
test_pos = split_edge['test']['edge'].to(device)
test_neg = split_edge['test']['edge_neg'].to(device)

# Hyperparameters
IN_CHANNELS = 2048  # Morgan fingerprint dimension
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.0
DECODER_DROPOUT = 0.0
EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 50000
EVAL_BATCH_SIZE = 50000
EVAL_EVERY = 5

# Create evaluation function wrapper
def evaluate_fn(model, pos_edges, neg_edges, batch_size):
    return evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size)

# Results storage
results = {}

# Train Morgan-GCN
logger.info("\n" + "=" * 80)
logger.info("Training Morgan-GCN Baseline")
logger.info("=" * 80)
gcn_model = MorganGCN(
    in_channels=IN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT
)
gcn_result = train_minimal_baseline(
    "Morgan-GCN",
    gcn_model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    eval_every=EVAL_EVERY,
    eval_batch_size=EVAL_BATCH_SIZE
)
results['Morgan-GCN'] = gcn_result

# Train Morgan-GraphSAGE
logger.info("\n" + "=" * 80)
logger.info("Training Morgan-GraphSAGE Baseline")
logger.info("=" * 80)
sage_model = MorganGraphSAGE(
    in_channels=IN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT
)
sage_result = train_minimal_baseline(
    "Morgan-GraphSAGE",
    sage_model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    eval_every=EVAL_EVERY,
    eval_batch_size=EVAL_BATCH_SIZE
)
results['Morgan-GraphSAGE'] = sage_result

# Train Morgan-GraphTransformer
logger.info("\n" + "=" * 80)
logger.info("Training Morgan-GraphTransformer Baseline")
logger.info("=" * 80)
transformer_model = MorganGraphTransformer(
    in_channels=IN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    heads=4,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT
)
transformer_result = train_minimal_baseline(
    "Morgan-GraphTransformer",
    transformer_model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=0.005,  # Lower LR for transformer
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    eval_every=EVAL_EVERY,
    eval_batch_size=EVAL_BATCH_SIZE
)
results['Morgan-GraphTransformer'] = transformer_result

# Train Morgan-GAT
logger.info("\n" + "=" * 80)
logger.info("Training Morgan-GAT Baseline")
logger.info("=" * 80)
gat_model = MorganGAT(
    in_channels=IN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    heads=4,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT
)
gat_result = train_minimal_baseline(
    "Morgan-GAT",
    gat_model,
    data,
    train_pos,
    valid_pos,
    valid_neg,
    test_pos,
    test_neg,
    evaluate_fn,
    device=device,
    epochs=EPOCHS,
    lr=0.005,  # Lower LR for attention
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    eval_every=EVAL_EVERY,
    eval_batch_size=EVAL_BATCH_SIZE
)
results['Morgan-GAT'] = gat_result

# Final results summary
logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS - MORGAN BASELINES")
logger.info("=" * 80)
for model_name, result in results.items():
    logger.info(f"{model_name}:")
    logger.info(f"  Validation Hits@20: {result.best_val_hits:.4f}")
    logger.info(f"  Test Hits@20: {result.best_test_hits:.4f}")
    val_test_gap = result.best_val_hits - result.best_test_hits
    gap_pct = (val_test_gap / result.best_val_hits * 100) if result.best_val_hits != 0 else 0.0
    logger.info(f"  Val-Test Gap: {val_test_gap:.4f} ({gap_pct:.1f}% relative)")
logger.info("=" * 80)
logger.info(f"Results logged to: {log_filename}")

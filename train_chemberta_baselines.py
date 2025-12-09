"""
Train ChemBERTa-based baseline models (GCN, GraphSAGE, GraphTransformer, GAT)

This script uses ChemBERTa embeddings (768-dim) as input features instead of learnable embeddings.
"""
import torch
import logging
import os
from datetime import datetime

from src.data import load_dataset_chemberta
from src.models.chemberta_baselines import ChemBERTaGCN, ChemBERTaGraphSAGE, ChemBERTaGraphTransformer, ChemBERTaGAT
from src.training import train_minimal_baseline
from src.evals import evaluate

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'logs/chemberta_baselines_{timestamp}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{log_dir}/chemberta_baselines.log'

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

# Load dataset with ChemBERTa embeddings
SMILES_CSV_PATH = "data/smiles.csv"  # CSV with columns: ogb_id, smiles
FEATURE_CACHE_PATH = "data/chemberta_features_768.pt"  # Cache file for faster loading
CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"  # HuggingFace model name

logger.info("Loading dataset with ChemBERTa embeddings...")
data, split_edge, num_nodes, evaluator = load_dataset_chemberta(
    'ogbl-ddi',
    device=device,
    smiles_csv_path=SMILES_CSV_PATH,
    feature_cache_path=FEATURE_CACHE_PATH,
    chemberta_model=CHEMBERTA_MODEL,
    batch_size=32
)

logger.info(f"Dataset loaded with ChemBERTa features: data.x.shape = {data.x.shape}")

train_pos = split_edge['train']['edge'].to(device)
valid_pos = split_edge['valid']['edge'].to(device)
valid_neg = split_edge['valid']['edge_neg'].to(device)
test_pos = split_edge['test']['edge'].to(device)
test_neg = split_edge['test']['edge_neg'].to(device)

# Hyperparameters
IN_CHANNELS = 768  # ChemBERTa embedding dimension
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

# Train ChemBERTa-GCN
logger.info("\n" + "=" * 80)
logger.info("Training ChemBERTa-GCN Baseline")
logger.info("=" * 80)
gcn_model = ChemBERTaGCN(
    in_channels=IN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT
)
gcn_result = train_minimal_baseline(
    "ChemBERTa-GCN",
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
results['ChemBERTa-GCN'] = gcn_result

# Train ChemBERTa-GraphSAGE
logger.info("\n" + "=" * 80)
logger.info("Training ChemBERTa-GraphSAGE Baseline")
logger.info("=" * 80)
sage_model = ChemBERTaGraphSAGE(
    in_channels=IN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT
)
sage_result = train_minimal_baseline(
    "ChemBERTa-GraphSAGE",
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
results['ChemBERTa-GraphSAGE'] = sage_result

# Train ChemBERTa-GraphTransformer
logger.info("\n" + "=" * 80)
logger.info("Training ChemBERTa-GraphTransformer Baseline")
logger.info("=" * 80)
transformer_model = ChemBERTaGraphTransformer(
    in_channels=IN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    heads=4,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT
)
transformer_result = train_minimal_baseline(
    "ChemBERTa-GraphTransformer",
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
results['ChemBERTa-GraphTransformer'] = transformer_result

# Train ChemBERTa-GAT
logger.info("\n" + "=" * 80)
logger.info("Training ChemBERTa-GAT Baseline")
logger.info("=" * 80)
gat_model = ChemBERTaGAT(
    in_channels=IN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    heads=4,
    dropout=DROPOUT,
    decoder_dropout=DECODER_DROPOUT
)
gat_result = train_minimal_baseline(
    "ChemBERTa-GAT",
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
results['ChemBERTa-GAT'] = gat_result

# Final results summary
logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS - CHEMBERTA BASELINES")
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

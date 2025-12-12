"""
Train Hybrid-GCN-Simple and log the learned α parameter
"""
import os
import logging
import torch
from datetime import datetime

from src.data.data_loader import load_dataset_chemberta
from src.models.hybrid import HybridGCN
from src.training.hybrid_trainer import train_hybrid_model

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/hybrid_simple_alpha_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "hybrid_simple_alpha.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
logger.info(f"Logging results to: {log_file}")

# Load dataset
logger.info("Loading dataset with ChemBERTa embeddings + SMILES mask...")
data, split_edge, num_nodes, evaluator = load_dataset_chemberta(
    dataset_name="ogbl-ddi",
    device=device,
    smiles_csv_path="data/smiles.csv",
    feature_cache_path="data/chemberta_features_768.pt",
    chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
    batch_size=32,
)

logger.info(f"Dataset loaded: {data.x.shape}, mask: {data.smiles_mask.sum():.0f}/{data.num_nodes}")

# Model
name = 'Hybrid-GCN-Simple-with-Alpha-Logging'
logger.info(f"\nTraining {name}")

model = HybridGCN(
    num_nodes=data.num_nodes,
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    chemical_dim=768,
    decoder_type='simple',  # Simple decoder with α
    decoder_dropout=0.3,
    use_gating=False,
).to(device)

logger.info(f"Model: {model.description}")
logger.info(f"Initial α value: {model.decoder.alpha.item():.6f}")

# Custom training loop to log α
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
train_edge_index = data.edge_index.to(device)
chemistry = data.x.to(device)
smiles_mask = data.smiles_mask.to(device)
train_pos = split_edge['train']['edge'].to(device)

logger.info("\nEpoch | Loss   | α value | Change")
logger.info("-" * 45)

batch_size = 5000
epochs = 50  # Just 50 epochs to see α evolution quickly
best_alpha = model.decoder.alpha.item()

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Encode
    z = model.encode(train_edge_index)
    
    # Sample negatives
    neg_edges = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=train_pos.size(0),
        method="sparse",
    ).t().to(device)
    
    # Train in batches
    num_batches = (train_pos.size(0) + batch_size - 1) // batch_size
    
    for batch_idx, start in enumerate(range(0, train_pos.size(0), batch_size)):
        end = min(start + batch_size, train_pos.size(0))
        pos_batch = train_pos[start:end]
        neg_batch = neg_edges[start:end]
        
        pos_logits = model.decode(z, chemistry, pos_batch, smiles_mask)
        neg_logits = model.decode(z, chemistry, neg_batch, smiles_mask)
        
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        batch_loss = (pos_loss + neg_loss) / num_batches
        
        batch_loss.backward(retain_graph=(batch_idx < num_batches - 1))
    
    optimizer.step()
    
    # Log α value
    current_alpha = model.decoder.alpha.item()
    alpha_change = current_alpha - best_alpha if epoch > 1 else 0
    best_alpha = current_alpha
    
    if epoch % 5 == 0 or epoch == 1:
        logger.info(f"{epoch:5d} | {(pos_loss + neg_loss).item():.4f} | {current_alpha:.6f} | {alpha_change:+.6f}")
    
    del z, neg_edges
    torch.cuda.empty_cache()

logger.info("-" * 45)
logger.info(f"\nFinal α value: {model.decoder.alpha.item():.6f}")
logger.info(f"Started at: 0.100000")
logger.info(f"Change: {model.decoder.alpha.item() - 0.1:+.6f} ({100*(model.decoder.alpha.item() - 0.1)/0.1:+.1f}%)")

# Save model
model_path = os.path.join(log_dir, "model_simple.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'alpha': model.decoder.alpha.item(),
}, model_path)
logger.info(f"\nModel saved to: {model_path}")
logger.info(f"Results logged to: {log_file}")


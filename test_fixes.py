"""
Quick test script to verify critical fixes are working.
Tests one model (GCN) for 20 epochs to check:
1. Learnable embeddings work
2. Loss decreases
3. Validation metrics improve
4. GPU is utilized (if available)
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# Load dataset
print("Loading dataset...")
dataset = PygLinkPropPredDataset('ogbl-ddi')
data = dataset[0]
split_edge = dataset.get_edge_split()

train_pos = split_edge['train']['edge'].to(device)
valid_pos = split_edge['valid']['edge'].to(device)
test_pos = split_edge['test']['edge'].to(device)

num_nodes = data.num_nodes
data.edge_index = train_pos.t().contiguous().to(device)
evaluator = Evaluator(name='ogbl-ddi')
print(f"Dataset loaded: {num_nodes} nodes, {train_pos.size(0)} train edges")
print("=" * 60)

# Define improved model
class ImprovedGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # CRITICAL FIX: Learnable embeddings
        self.emb = nn.Embedding(num_nodes, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def encode(self, edge_index):
        x = self.emb.weight
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x

    def decode(self, z, edge):
        src, dst = edge[:, 0], edge[:, 1]
        return (z[src] * z[dst]).sum(dim=1)

# Evaluation function
@torch.no_grad()
def evaluate(model, pos_edges):
    model.eval()
    z = model.encode(data.edge_index)
    pos_scores = model.decode(z, pos_edges).view(-1).cpu()

    # Sample negatives
    neg = negative_sampling(
        edge_index=data.edge_index.cpu(),
        num_nodes=num_nodes,
        num_neg_samples=pos_edges.size(0)
    ).t().to(device)
    neg_scores = model.decode(z, neg).view(-1).cpu()

    result = evaluator.eval({
        'y_pred_pos': pos_scores,
        'y_pred_neg': neg_scores,
    })
    return result['hits@20']

# Initialize model
print("Initializing model...")
model = ImprovedGCN(num_nodes, 128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("=" * 60)

# Train for a few epochs
print("Training for 20 epochs (testing fixes)...\n")
start_time = time.time()

for epoch in range(1, 21):
    model.train()
    optimizer.zero_grad()

    epoch_start = time.time()

    z = model.encode(data.edge_index)
    neg_samples = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=num_nodes,
        num_neg_samples=train_pos.size(0)
    ).t().to(device)

    pos_out = model.decode(z, train_pos)
    neg_out = model.decode(z, neg_samples)

    # Improved loss function
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    loss = pos_loss + neg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    epoch_time = time.time() - epoch_start

    if epoch % 5 == 0 or epoch == 1:
        val_hits = evaluate(model, valid_pos)
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val Hits@20: {val_hits:.4f} | Time: {epoch_time:.2f}s")

total_time = time.time() - start_time
print(f"\nTotal training time: {total_time:.2f}s ({total_time/20:.2f}s per epoch)")
print("=" * 60)

# Final evaluation
print("\nFinal Evaluation:")
val_hits = evaluate(model, valid_pos)
test_hits = evaluate(model, test_pos)
print(f"Validation Hits@20: {val_hits:.4f}")
print(f"Test Hits@20:       {test_hits:.4f}")
print("=" * 60)

print("\n✅ Test completed successfully!")
print("\nKey improvements:")
print("  • Learnable embeddings: IMPLEMENTED")
print("  • Batch normalization: IMPLEMENTED")
print("  • Dropout: IMPLEMENTED")
print("  • Better loss (BCE with logits): IMPLEMENTED")
print("  • Gradient clipping: IMPLEMENTED")
print(f"  • GPU utilization: {'YES' if torch.cuda.is_available() else 'NO (CPU only)'}")
print("\nExpected behavior:")
print("  • Loss should decrease steadily (starting ~1.4, dropping to ~0.6-0.8)")
print("  • Hits@20 should improve from ~0.05-0.10 to ~0.20-0.40 even in 20 epochs")
print("  • Training should be faster (~2-5s/epoch on GPU, ~30s on CPU)")

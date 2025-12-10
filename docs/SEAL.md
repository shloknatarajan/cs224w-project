# SEAL: Subgraph Extraction and Labeling for Link Prediction

## Overview

SEAL (Subgraph Extraction and Labeling) is a state-of-the-art approach for link prediction that differs fundamentally from traditional GNN-based methods. Instead of learning node embeddings on the entire graph and computing scores between node pairs, SEAL extracts local subgraphs around each potential link and learns to classify them directly.

**Reference Paper:**
> M. Zhang and Y. Chen. "Link Prediction Based on Graph Neural Networks." NeurIPS 2018.
> https://arxiv.org/abs/1802.09691

## Key Concepts

### 1. Subgraph Extraction

For each candidate link (u, v), SEAL extracts a k-hop enclosing subgraph that includes:
- All nodes within k hops of u
- All nodes within k hops of v
- All edges between these nodes (excluding the target link if it exists)

This local subgraph captures the structural context around the potential link.

### 2. DRNL Node Labeling

SEAL uses **Double Radius Node Labeling (DRNL)** to encode the structural role of each node in the subgraph relative to the target link endpoints:

- **Label 1**: The two endpoints (u and v) of the target link
- **Label 2+**: Other nodes based on their shortest path distances to u and v

The DRNL formula ensures that:
- Nodes at the same distance from both endpoints get the same label
- Nodes at different distances get different labels
- The labeling is invariant to graph isomorphism

### 3. GNN Classification

A GNN (typically GCN or GIN) processes the labeled subgraph and produces a binary classification:
- **1**: Link exists
- **0**: Link does not exist

The model learns structural patterns that indicate link formation.

## Architecture

SEAL consists of three main components:

1. **Node Embedding Layer**: Embeds DRNL labels into continuous vectors
2. **GNN Layers**: Multiple GCN or GIN layers to learn from subgraph structure
3. **Graph Pooling**: Aggregates node representations into a graph-level representation
   - **SortPool** (recommended): Sorts nodes and takes top-k
   - **Add/Mean/Max Pool**: Standard graph pooling methods
4. **MLP Classifier**: Final classification layers

## Implementation

### File Structure

```
src/
├── data/
│   └── seal_utils.py          # Subgraph extraction and DRNL labeling
└── models/
    └── advanced/
        └── seal.py             # SEAL-GCN and SEAL-GIN models

train_seal.py                   # Training script
test_seal.py                    # Unit tests
```

### Models

#### SEAL-GCN
Uses Graph Convolutional Network layers for learning from subgraphs.

```python
from src.models.advanced import SEALGCN

model = SEALGCN(
    max_z=1000,          # Maximum DRNL label value
    hidden_dim=32,       # Hidden dimension
    num_layers=3,        # Number of GNN layers
    k=30,               # SortPool k value
    dropout=0.5,        # Dropout rate
    pooling='sort'      # Pooling method
)
```

#### SEAL-GIN
Uses Graph Isomorphism Network layers for more expressive power.

```python
from src.models.advanced import SEALGIN

model = SEALGIN(
    max_z=1000,
    hidden_dim=32,
    num_layers=3,
    k=30,
    dropout=0.5,
    pooling='sort'
)
```

### Data Processing

#### SEALDataset
Creates a PyTorch Dataset that extracts subgraphs on-the-fly.

```python
from src.data import SEALDataset, seal_collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = SEALDataset(
    edge_index=train_edge_index,  # Training graph edges
    pos_edges=positive_edges,      # Positive examples
    neg_edges=negative_edges,      # Negative examples
    num_nodes=num_nodes,
    num_hops=1,                    # k-hop subgraphs
    max_nodes_per_hop=None         # Limit nodes (optional)
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=seal_collate_fn
)
```

#### Manual Subgraph Extraction

```python
from src.data import extract_enclosing_subgraph, drnl_node_labeling

# Extract subgraph for link (src, dst)
subgraph = extract_enclosing_subgraph(
    edge_index=edge_index,
    src=0,
    dst=5,
    num_nodes=num_nodes,
    num_hops=1
)

# subgraph.x contains DRNL labels
# subgraph.edge_index contains subgraph edges
```

## Training

### Basic Training Script

```bash
# Train SEAL-GIN model
pixi run python train_seal.py
```

### Configuration

Key hyperparameters in `train_seal.py`:

```python
NUM_HOPS = 1              # Subgraph extraction radius
HIDDEN_DIM = 32           # Model hidden dimension
NUM_LAYERS = 3            # Number of GNN layers
K = 30                    # SortPool k value
DROPOUT = 0.5             # Dropout rate
POOLING = 'sort'          # Pooling method
BATCH_SIZE = 32           # Subgraph batch size
LEARNING_RATE = 0.0001    # Learning rate
MODEL_TYPE = 'GIN'        # 'GCN' or 'GIN'
```

### Training Loop

SEAL uses binary cross-entropy loss:

```python
# Forward pass
out = model(batch)  # batch contains multiple subgraphs

# Loss
loss = F.binary_cross_entropy_with_logits(out, batch.y)

# Backward pass
loss.backward()
optimizer.step()
```

## Evaluation

Unlike baseline GNN models that use Hits@K metrics, SEAL typically reports:
- **Accuracy**: Percentage of correctly classified links
- **AUC**: Area Under ROC Curve
- **AP**: Average Precision

```python
def evaluate_seal(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = (out > 0).float()
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total
```

## Advantages and Disadvantages

### Advantages
- **Structure-aware**: Directly learns from local graph structure
- **No feature engineering**: DRNL labels capture structural role automatically
- **Proven effectiveness**: State-of-the-art results on many benchmarks
- **Generalizable**: Works well across different graph types

### Disadvantages
- **Computational cost**: Must extract and process subgraphs for each link
- **Scalability**: Can be slow for very large graphs or dense evaluation sets
- **Memory intensive**: Stores multiple subgraphs in memory during training
- **Inference complexity**: O(k × |E|) where k is number of links to predict

## Optimization Tips

### 1. Limit Evaluation Set Size
For large graphs, sample a subset of validation/test edges:

```python
MAX_EVAL_SAMPLES = 5000
valid_pos_sample = valid_pos[:MAX_EVAL_SAMPLES]
valid_neg_sample = valid_neg[:MAX_EVAL_SAMPLES]
```

### 2. Adjust k-hop Parameter
- **k=1**: Faster, works well for local structure
- **k=2+**: More context, but much slower

### 3. Use max_nodes_per_hop
Limit subgraph size for very dense graphs:

```python
dataset = SEALDataset(
    edge_index, pos_edges, neg_edges, num_nodes,
    num_hops=1,
    max_nodes_per_hop=50  # Limit to 50 nodes per hop
)
```

### 4. Batch Size
- Smaller batches: More memory-efficient
- Larger batches: Faster training (if memory allows)

## Comparison with Baseline GNNs

| Aspect | Baseline GNN (GCN/SAGE/GAT) | SEAL |
|--------|------------------------------|------|
| **Approach** | Global node embeddings + decoder | Local subgraph classification |
| **Features** | Node embeddings (learnable) | DRNL structural labels |
| **Training** | Full graph forward pass | Subgraph batches |
| **Inference** | Fast (dot product) | Slower (subgraph extraction) |
| **Memory** | Graph-level (O(V + E)) | Batch-level (O(batch × subgraph)) |
| **Metrics** | Hits@K | Accuracy/AUC |
| **Best for** | Large-scale, fast inference | Structural pattern learning |

## Results on OGBL-DDI

Expected performance on OGBL-DDI dataset:

- **SEAL-GIN**: ~80-85% accuracy on sampled validation set
- **SEAL-GCN**: ~75-80% accuracy on sampled validation set

Note: Direct comparison with baseline Hits@20 metrics is not straightforward due to different evaluation protocols.

## Testing

Run the test suite:

```bash
pixi run python test_seal.py
```

This tests:
- DRNL node labeling
- Subgraph extraction
- SEALDataset functionality
- Model forward passes
- Different pooling methods

## References

1. **Original SEAL Paper**:
   - M. Zhang and Y. Chen. "Link Prediction Based on Graph Neural Networks." NeurIPS 2018.
   - https://arxiv.org/abs/1802.09691

2. **PyTorch Geometric Documentation**:
   - https://pytorch-geometric.readthedocs.io/

3. **Open Graph Benchmark (OGB)**:
   - https://ogb.stanford.edu/

## Future Improvements

Potential enhancements:
1. **Parallel subgraph extraction**: Use multiprocessing for faster data loading
2. **Caching**: Pre-compute and cache subgraphs for training efficiency
3. **Adaptive k-hop**: Learn optimal subgraph radius per node pair
4. **Feature augmentation**: Combine DRNL labels with node features
5. **Attention mechanisms**: Use graph attention in pooling layer

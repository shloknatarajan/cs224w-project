# Node2Vec-Enhanced GNN Models

Node2Vec-enhanced versions of standard GNN architectures for link prediction on ogbl-ddi.

## Overview

These models combine **unsupervised Node2Vec pretraining** with **supervised GNN training**:

1. **Pretraining**: Node2Vec learns structural embeddings via random walks
2. **Initialization**: GNN models are initialized with Node2Vec embeddings  
3. **Fine-tuning**: Embeddings are fine-tuned end-to-end during link prediction

## Key Insight

**Frozen Node2Vec embeddings hurt performance** (test Hits@20: 0.0712)  
**Fine-tuned Node2Vec embeddings improve performance** (test Hits@20: 0.1370)

The embeddings need to adapt to the link prediction objective to be effective.

## Architecture

All models follow this pattern:

```python
# Input features: Concatenate learnable IDs + Node2Vec embeddings
x = torch.cat([id_embeddings, node2vec_embeddings], dim=1)
x = linear_projection(x)  # Project to hidden_dim

# GNN layers (GCN/SAGE/Transformer/GAT)
for layer in gnn_layers:
    x = layer(x, edge_index)
    x = relu(x)

# Decode (dot product or MLP)
score = decode(x_i, x_j)
```

## Models

### Node2VecGCN
- **Architecture**: Graph Convolutional Network
- **Best Baseline**: 0.1237 test Hits@20
- **With Node2Vec**: 0.1370 test Hits@20 ✅ (+10.8%)
- **Status**: Best performing model

### Node2VecGraphSAGE
- **Architecture**: GraphSAGE with mean aggregation
- **Best Baseline**: 0.0365 test Hits@20
- **With Node2Vec**: TBD
- **Notes**: Baseline was weak, Node2Vec might help

### Node2VecTransformer
- **Architecture**: Graph Transformer with attention
- **Best Baseline**: 0.0439 test Hits@20
- **With Node2Vec**: TBD
- **Notes**: Multi-head attention + Node2Vec structure

### Node2VecGAT
- **Architecture**: Graph Attention Network
- **Best Baseline**: 0.0503 test Hits@20
- **With Node2Vec**: TBD
- **Notes**: Attention mechanisms + unsupervised priors

## Usage

### Train All Models

```bash
pixi run train-node2vec
```

This runs all four Node2Vec-enhanced models with default hyperparameters.

### Train Individual Model

```python
from src.models.node2vec import Node2VecGCN
from src.training.node2vec_trainer import train_node2vec_embeddings, Node2VecConfig

# Pretrain Node2Vec
config = Node2VecConfig(
    dim=64,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    epochs=50,
)
embeddings = train_node2vec_embeddings(edge_index, num_nodes, config, device)

# Create model
model = Node2VecGCN(
    node2vec_embeddings=embeddings,
    hidden_dim=128,
    num_layers=2,
)

# Train (embeddings will be fine-tuned)
result = train_minimal_baseline(...)
```

### Custom Hyperparameters

```bash
# More Node2Vec pretraining
pixi run python train_node2vec_baselines.py --node2vec-epochs 100

# Smaller Node2Vec dimension
pixi run python train_node2vec_baselines.py --node2vec-dim 32

# Different GNN architecture
pixi run python train_node2vec_baselines.py --num-layers 3 --hidden-dim 256
```

## Hyperparameters

### Node2Vec Pretraining
- `node2vec_dim`: 64 (embedding dimension)
- `walk_length`: 20 (random walk length)
- `context_size`: 10 (context window for skip-gram)
- `walks_per_node`: 10 (number of walks starting from each node)
- `epochs`: 50 (pretraining epochs - more is better!)

### GNN Training
- `hidden_dim`: 128
- `num_layers`: 2
- `lr`: 0.01 (GCN/SAGE), 0.005 (Transformer/GAT)
- `weight_decay`: 1e-4
- `dropout`: 0.0 (disabled per baseline requirements)
- `epochs`: 100
- `batch_size`: 50000

## Results

| Model | Val Hits@20 | Test Hits@20 | Gap |
|-------|-------------|--------------|-----|
| **Vanilla Baselines** |
| GCN | 0.1470 | 0.1237 | 15.8% |
| GraphSAGE | 0.1090 | 0.0365 | 66.5% |
| Transformer | 0.0907 | 0.0439 | 51.6% |
| GAT | 0.1864 | 0.0503 | 73.0% |
| **Node2Vec Enhanced** |
| Node2Vec-GCN | 0.1852 | **0.1370** ✅ | 26.0% |
| Node2Vec-SAGE | TBD | TBD | TBD |
| Node2Vec-Transformer | TBD | TBD | TBD |
| Node2Vec-GAT | TBD | TBD | TBD |

## Why Does This Work?

1. **Complementary signals**: Node2Vec captures global structure, GNN learns local patterns
2. **Better initialization**: Starting from random walks is better than random
3. **Task adaptation**: Fine-tuning lets embeddings adjust to link prediction
4. **Regularization**: Node2Vec acts as a structural prior that reduces overfitting

## Files

```
src/models/node2vec/
├── __init__.py          # Exports
├── README.md           # This file
├── gcn.py              # Node2VecGCN
├── sage.py             # Node2VecGraphSAGE
├── transformer.py      # Node2VecTransformer
└── gat.py              # Node2VecGAT

train_node2vec_baselines.py  # Main training script
train_node2vec_gcn.py         # Single GCN training (legacy)
```

## Design Principles

1. **Zero dropout**: Following baseline requirements (dropout must be 0.0)
2. **Fine-tunable embeddings**: Node2Vec embeddings are NOT frozen
3. **Concatenation**: Combine Node2Vec + learnable IDs for richer features
4. **Simple decoders**: Dot product or MLP, no fancy chemistry-aware decoders
5. **Consistent architecture**: All models follow the same pattern

## Future Work

- Experiment with freezing vs. fine-tuning at different layers
- Try different Node2Vec dimensions (32, 128, 256)
- Test deeper random walks (walk_length=40, 80)
- Combine with Morgan fingerprints or ChemBERTa
- Layer-wise learning rates (lower LR for Node2Vec, higher for GNN)


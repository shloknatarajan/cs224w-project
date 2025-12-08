# Training Analysis - Advanced GCN (20251208_031940)

## Performance Summary
- **Best Validation Hits@20**: 0.0170 (1.7%) at epoch 115
- **Test Hits@20 at best**: 0.0249 (2.49%)
- **Final Status**: Training likely stopped early or still running (last log: epoch 280)

## Critical Issues

### 1. Gradient Vanishing (SEVERE)
**Symptom**: Gradient norm = 0.00 from epoch 5 onwards
**Impact**: Model cannot learn effectively; weights barely update
**Root Causes**:
- Batch normalization + high dropout (0.5) combination
- 3-layer GCN may be over-smoothing
- BCE loss with hard negatives causing saturation

**Solutions**:
- Remove or reduce batch normalization
- Lower dropout to 0.2-0.3
- Switch to margin-based loss (e.g., triplet loss, AUC loss)
- Add gradient clipping
- Use LayerNorm instead of BatchNorm
- Reduce model depth to 2 layers

### 2. Embedding Collapse
**Symptom**: High mean similarity (0.4-0.8), low std (0.22-0.24)
**Impact**: All nodes learn similar representations; cannot distinguish edges
**Solutions**:
- Add embedding regularization (diversity loss)
- Increase hidden dimension to 256
- Use contrastive learning objectives
- Add node feature augmentation
- Consider VGAE instead of deterministic encoder

### 3. Poor Link Prediction Performance
**Symptom**: Only 1.7% validation accuracy after 115 epochs
**Comparison**: State-of-the-art on ogbl-ddi achieves 60-90% Hits@20
**Gap**: ~58-88% below competitive baselines

**Fundamental Issues**:
- Architecture too simple for drug-drug interaction complexity
- Missing critical components:
  - Multi-hop neighborhood aggregation
  - Attention mechanisms
  - Edge features/attributes
  - Graph-level pooling

**Alternative Approaches**:
- Use SEAL (Subgraph Extraction And Labeling)
- Implement Graph Attention Networks (GAT)
- Try NGCF or LightGCN architectures
- Use pre-trained molecular embeddings

### 4. Hard Negative Sampling Not Effective
**Observation**: Switched to hard negatives at epoch 55, no improvement
**Possible Reasons**:
- Hard negatives too hard (model already struggling)
- Ratio of 0.05 may be too low
- Model embeddings already collapsed before hard neg sampling

**Solutions**:
- Start hard negatives earlier (epoch 10-20)
- Gradually increase hard negative ratio: 0.0 → 0.1 → 0.3 → 0.5
- Use semi-hard negatives (moderate difficulty)
- Implement curriculum learning

### 5. Training Inefficiency
**Issues**:
- No improvement after epoch 115 (165 wasted epochs)
- Learning rate decay not helping (0.003 → 0.0015 → 0.00075)
- Early stopping patience too high (40 epochs)

**Solutions**:
- Reduce patience to 20 epochs
- Use cosine annealing LR schedule
- Implement warm restarts
- Add validation-based LR reduction (ReduceLROnPlateau)

## Structural Feature Analysis
**Features Used**: Degree (6 dims total)
- Degree: mean=4.47, std=1.89
- Clustering: mean=0.51, std=0.22
- Core number: mean=5.12, std=1.54
- PageRank: mean=0.11, std=0.08

**Issues**:
- Only 6 structural dimensions (very low)
- Features not well-scaled (different ranges)
- Missing graph-level features

**Improvements**:
- Add more features: betweenness, closeness, eigenvector centrality
- Include node2vec or random walk embeddings
- Add local subgraph features (triangle count, ego-graph density)
- Scale features properly (StandardScaler)

## Model Architecture Issues

**Current**: 3-layer GCN, hidden_dim=192, dropout=0.5
**Problems**:
- Over-smoothing after 3 hops
- Hidden dim too small for 4267 nodes
- Simple decoder (dot product) insufficient

**Recommended Architecture**:
```
Encoder:
  - 2-layer GAT (attention_heads=4, hidden=256)
  - Skip connections between all layers
  - LayerNorm (not BatchNorm)
  - Dropout=0.2

Decoder:
  - MLP with 2 hidden layers [512, 256]
  - Combine: concat(src, dst, hadamard, L1, L2)
  - Add edge-level attention
```

## Immediate Action Items

### High Priority
1. Fix gradient vanishing:
   - Remove BatchNorm or replace with LayerNorm
   - Reduce dropout to 0.2
   - Add gradient clipping (max_norm=1.0)

2. Improve loss function:
   - Switch to AUCLoss or InfoNCE
   - Add margin-based constraints

3. Reduce model depth:
   - Try 2 layers first
   - Add skip connections

### Medium Priority
4. Enhance decoder:
   - Use MLP decoder with multiple predictor types
   - Combine dot product, hadamard, concat, L1/L2

5. Better negative sampling:
   - Start hard negatives at epoch 10
   - Use curriculum strategy

6. Add regularization:
   - Embedding diversity loss
   - L2 regularization on embeddings

### Low Priority
7. Hyperparameter tuning:
   - Hidden dim: try 256, 384
   - Learning rate: try 0.001, 0.005
   - Batch size: try 32k, 65k

8. Advanced techniques:
   - Node feature dropout
   - DropEdge during training
   - Virtual nodes

## Expected Improvements
With fixes above:
- **Gradient health**: Expect grad_norm > 0.1
- **Validation Hits@20**: Target 15-30% (10-18x improvement)
- **Training stability**: Should see steady improvement for 50+ epochs
- **Embedding quality**: Similarity std > 0.3, more diverse representations

## Alternative: Try Different Baseline
Consider implementing:
1. **SEAL** (best for link prediction): Uses subgraph patterns
2. **BUDDY** (state-of-art on ogbl-ddi): Reaches 80%+ Hits@20
3. **Neo-GNN**: Handles hard negatives better
4. **PLNLP**: Pairwise learning, good for drug interactions

## References for ogbl-ddi
- Current model: 1.7% Hits@20
- Simple heuristics (Common Neighbors): ~10-15%
- GCN baseline (proper): ~20-30%
- GAT/GraphSAINT: ~40-50%
- SEAL/BUDDY/SOTA: ~60-90%

**Gap to close**: Need 10-50x improvement to be competitive.

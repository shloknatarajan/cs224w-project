# Model Results Summary

**Last Updated**: December 10, 2025
**Dataset**: ogbl-ddi (4267 nodes, 2.1M edges)
**Metric**: Hits@20 (primary), Hits@10, Hits@30 (supplementary)

---

## Best Performing Models

### 1. GCN with All External Features ⭐ **BEST MODEL**
**Log**: `ddi_gcn_all_20251210_033923/ddi_gcn_all.log`
**Date**: December 10, 2025

| Metric | Validation | Test | Best Epoch |
|--------|-----------|------|------------|
| **Hits@10** | - | **35.29%** | 1685 |
| **Hits@20** | **70.31%** | **73.28%** | 1685 |
| **Hits@30** | - | **84.64%** | 1685 |

**Configuration**:
- Architecture: GCN (2 layers) + MLP decoder
- Hidden channels: 256
- Dropout: 0.5
- Learning rate: 0.005
- Batch size: 65,536
- External features (3054 dims total):
  - Morgan fingerprints: 2048 dims
  - PubChem properties: 9 dims
  - ChemBERTa embeddings: 768 dims
  - Drug-target features: 229 dims

**Key Observations**:
- Excellent generalization: Test performance (73.28%) exceeds validation (70.31%)
- Long training: Best results at epoch 1685 out of 2000
- Feature engineering crucial: Combined external features provide rich drug representations

---

## Experimental Models

### 2. Hybrid GCN (Quick Tuning)
**Log**: `hybrid_gcn_quick_tuning_20251210_015911/quick_tuning.log`
**Date**: December 10, 2025

**Best Configuration** (Experiment 3):
| Metric | Validation | Test | Best Epoch |
|--------|-----------|------|------------|
| **Hits@20** | **22.82%** | **15.72%** | 40 |

**Configuration**:
- Hidden dim: 96
- Num layers: 2
- Decoder dropout: 0.4
- Learning rate: 0.01
- Weight decay: 0.0005
- ChemBERTa features: 768 dims
- Structural features: 6 dims (degree, clustering, core number, PageRank)

**Key Observations**:
- Adaptive α parameter: Started at 0.1, decreased to 0.0038 (-96.2%)
- 20 experiments conducted with 50 epochs each
- Significant val-test gap indicates potential overfitting

### 3. GDIN (Graph Deconfounded Learning)
**Log**: `gdin_20251209_233115/gdin.log`
**Date**: December 9, 2025

| Metric | Validation | Test | Best Epoch |
|--------|-----------|------|------------|
| **Hits@20** | **19.59%** | **10.95%** | 235 |

**Configuration**:
- Hidden dim: 256
- Num layers: 2
- Dropout: 0.0
- Learning rate: 0.005
- Num negatives: 3
- AUC Loss (Phase 1)

**Key Observations**:
- Large val-test gap: 8.65% (44.1% relative)
- Early stopping at epoch 385 (no improvement for 30 evaluations)
- Specialized architecture for causal deconfounding

### 4. Node2Vec + GCN
**Log**: `node2vec_gcn_20251209_225240/node2vec_gcn.log`
**Date**: December 9, 2025

| Metric | Validation | Test | Best Epoch |
|--------|-----------|------|------------|
| **Hits@20** | **16.62%** | **13.19%** | 70 |

**Configuration**:
- Node2Vec dim: 64
- Hidden dim: 128
- GCN layers: 2
- Decoder: simple
- Node2Vec training: 50 epochs (walk_length=20, context=10)

**Key Observations**:
- Good generalization: Val-test gap only 3.43% (20.6% relative)
- Pre-trained embeddings: Node2Vec captures structural patterns
- Fast convergence: Best results at epoch 70

---

## Baseline Models

### 5. Standard GNN Baselines (Graph Structure Only)
**Log**: `baselines_20251209_185813/baselines.log`
**Date**: December 9, 2025

| Model | Validation Hits@20 | Test Hits@20 | Best Epoch | Val-Test Gap |
|-------|-------------------|--------------|------------|--------------|
| **GCN** | 13.59% | 11.02% | 135 | 2.58% (19.0%) |
| **GAT** | 9.53% | 4.88% | 170 | 4.65% (48.8%) |
| **GraphSAGE** | 9.61% | 4.46% | 80 | 5.15% (53.6%) |
| **GraphTransformer** | 11.66% | 3.73% | 150 | 7.93% (68.0%) |

**Configuration** (all models):
- Hidden dim: 128
- Num layers: 2
- Dropout: 0.0
- Learning rate: 0.01 (0.005 for Transformer/GAT)
- Batch size: 50,000
- Input: Graph structure only (no node features)

**Key Observations**:
- GCN performs best among structure-only baselines
- Significant val-test gaps indicate overfitting
- GraphTransformer has largest gap (68%)
- Limited by lack of node features

### 6. Morgan Fingerprint Baselines
**Log**: `morgan_baselines_20251209_191058/morgan_baselines.log`
**Date**: December 9, 2025

| Model | Validation Hits@20 | Test Hits@20 | Best Epoch | Val-Test Gap |
|-------|-------------------|--------------|------------|--------------|
| **Morgan-GCN** | 6.10% | 5.36% | 185 | 0.73% (12.0%) |
| **Morgan-GAT** | 4.32% | 1.12% | 180 | 3.20% (74.0%) |
| **Morgan-GraphTransformer** | 4.14% | 3.98% | 185 | 0.15% (3.7%) |
| **Morgan-GraphSAGE** | 3.05% | 4.71% | 200 | -1.65% (-54.2%) |

**Configuration** (all models):
- Morgan fingerprints: 2048 dims
- Hidden dim: 128
- Num layers: 2
- Dropout: 0.0
- Learning rate: 0.01 (0.005 for Transformer/GAT)
- Batch size: 50,000

**Key Observations**:
- Lower performance than structure-only baselines
- Morgan features alone insufficient for this task
- GraphSAGE shows negative gap (test > val)
- GraphTransformer has best generalization (3.7% gap)

---

## Model Comparison

### Performance Ranking (Test Hits@20)

1. **GCN + All External Features**: 73.28% ⭐
2. **Hybrid GCN** (Exp 3): 15.72%
3. **Node2Vec + GCN**: 13.19%
4. **GCN Baseline**: 11.02%
5. **GDIN**: 10.95%
6. **Morgan-GCN**: 5.36%
7. **GraphSAGE Baseline**: 4.46%
8. **GAT Baseline**: 4.88%
9. **Morgan-GraphSAGE**: 4.71%
10. **Morgan-GraphTransformer**: 3.98%
11. **GraphTransformer Baseline**: 3.73%
12. **Morgan-GAT**: 1.12%

### Key Insights

1. **External Features Are Critical**:
   - GCN with all external features (73.28%) vastly outperforms GCN baseline (11.02%)
   - 6.6x improvement from adding molecular features

2. **Feature Quality Matters**:
   - Combined features (Morgan + PubChem + ChemBERTa + Drug-targets) work best
   - Morgan fingerprints alone perform worse than graph structure alone

3. **Generalization Patterns**:
   - Best model shows positive generalization (test > val)
   - Most experimental models show overfitting
   - Node2Vec + GCN has excellent generalization

4. **Architecture Insights**:
   - Simple GCN architecture works well with rich features
   - More complex architectures (GAT, Transformer) don't help without good features
   - Node embeddings (Node2Vec) provide complementary signal

5. **Training Considerations**:
   - Best model requires extended training (1685 epochs)
   - Most models converge within 200 epochs
   - Early stopping prevents catastrophic overfitting

---

## Dataset Information

- **Nodes**: 4,267 drugs
- **Train edges**: 1,067,911 positive
- **Valid edges**: 133,489 positive, 101,882 negative
- **Test edges**: 133,489 positive, 95,599 negative
- **Total edges** (with self-loops): 1,072,178

---

## Training Environment

- **Device**: CUDA (GPU)
- **Framework**: PyTorch Geometric
- **Evaluation Metric**: Hits@K (K=10, 20, 30)
- **Evaluation Strategy**: Track best validation performance, report test at that epoch

---

## Recommendations

### For Best Performance:
1. **Use rich external features**: Combine multiple feature sources (molecular, biological, structural)
2. **Start with GCN**: Simple architecture with good features outperforms complex architectures
3. **Train longer**: Best results often require >1000 epochs with proper monitoring
4. **Monitor generalization**: Val-test gap is crucial indicator

### For Future Experiments:
1. **Ensemble methods**: Combine GCN + All Features with Node2Vec + GCN
2. **Feature ablation**: Understand individual contribution of each feature type
3. **Attention mechanisms**: Apply to combined feature representation
4. **Graph augmentation**: Explore augmentation strategies for drug-drug graphs
5. **Complete Heuristic GCN**: Finish incomplete experiment with structural heuristics

---

## File Locations

All experiment logs are stored in: `/home/ubuntu/cs224w-project/logs/`

Key log files:
- Best model: `ddi_gcn_all_20251210_033923/ddi_gcn_all.log`
- Baselines: `baselines_20251209_185813/baselines.log`
- Morgan baselines: `morgan_baselines_20251209_191058/morgan_baselines.log`
- GDIN: `gdin_20251209_233115/gdin.log`
- Node2Vec: `node2vec_gcn_20251209_225240/node2vec_gcn.log`
- Hybrid GCN: `hybrid_gcn_quick_tuning_20251210_015911/quick_tuning.log`

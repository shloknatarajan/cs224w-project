# Morgan Baselines vs Minimal Trainer Baselines Analysis

**Date:** December 9, 2025  
**Dataset:** ogbl-ddi (4,267 drug nodes, ~1M training edges)  
**Metric:** Hits@20

## Executive Summary

The addition of Morgan fingerprint features (2048-dimensional) **significantly degraded performance** across all four GNN architectures compared to the minimal trainer baselines that used no node features. This is a critical finding that suggests the Morgan fingerprints may be introducing noise or that the models are failing to effectively leverage this chemical information.

---

## Results Comparison

### Performance Summary Table

| Model | Minimal Trainer Val@20 | Morgan Val@20 | Δ Val | Minimal Trainer Test@20 | Morgan Test@20 | Δ Test |
|-------|----------------------:|-------------:|------:|------------------------:|--------------:|-------:|
| **GCN** | 0.1118 | 0.0610 | **-45.4%** | 0.0995 | 0.0536 | **-46.1%** |
| **GraphSAGE** | 0.1293 | 0.0305 | **-76.4%** | 0.0594 | 0.0471 | **-20.7%** |
| **GraphTransformer** | 0.1283 | 0.0414 | **-67.7%** | 0.0590 | 0.0398 | **-32.5%** |
| **GAT** | 0.1190 | 0.0432 | **-63.7%** | 0.0728 | 0.0112 | **-84.6%** |

### Absolute Performance Degradation

**Validation Set:**
- Best minimal trainer model: GraphSAGE (0.1293)
- Best Morgan model: Morgan-GCN (0.0610)
- **Gap: 0.0683 (52.8% relative decrease)**

**Test Set:**
- Best minimal trainer model: GCN (0.0995)
- Best Morgan model: Morgan-GCN (0.0536)
- **Gap: 0.0459 (46.1% relative decrease)**

---

## Detailed Model-by-Model Analysis

### 1. GCN - Most Stable, Least Affected

**Minimal Trainer:**
- Validation: 0.1118 (epoch 180)
- Test: 0.0995
- Val-Test Gap: 0.0123 (11.0%)
- Training: Converged smoothly after ~100 epochs

**Morgan Features:**
- Validation: 0.0610 (epoch 185)
- Test: 0.0536
- Val-Test Gap: 0.0073 (12.0%)
- **Performance Drop: ~46% on both val and test**

**Analysis:**
- GCN showed the **smallest relative degradation** (~46%)
- Maintained similar val-test consistency (11% vs 12% gap)
- Simple aggregation in GCN may be more robust to noisy features
- Still converged, suggesting optimization was not the primary issue

### 2. GraphSAGE - Severe Validation Collapse

**Minimal Trainer:**
- Validation: 0.1293 (epoch 200)
- Test: 0.0594
- Val-Test Gap: 0.0700 (54.1%)
- Training: Best performance among minimal baselines

**Morgan Features:**
- Validation: 0.0305 (epoch 200)
- Test: 0.0471
- Val-Test Gap: -0.0165 (-54.2%, **test > val**)
- **Performance Drop: 76.4% on validation, 20.7% on test**

**Analysis:**
- Most dramatic performance degradation on validation set
- Unusual inverted gap (test outperformed validation)
- Suggests severe overfitting or learning pathology
- SAGE's neighborhood sampling may interact poorly with dense Morgan features
- The 2048-dimensional features may overwhelm the aggregation mechanism

### 3. GraphTransformer - Poor Feature Integration

**Minimal Trainer:**
- Validation: 0.1283 (epoch 105)
- Test: 0.0590
- Val-Test Gap: 0.0693 (54.0%)
- Training: Fast initial convergence

**Morgan Features:**
- Validation: 0.0414 (epoch 185)
- Test: 0.0398
- Val-Test Gap: 0.0015 (3.7%)
- **Performance Drop: 67.7% on validation, 32.5% on test**

**Analysis:**
- Second-worst degradation on validation
- Improved val-test consistency (54.0% → 3.7% gap)
- Suggests the model may be underfitting with Morgan features
- Attention mechanism may struggle with the high-dimensional, dense features
- Slower convergence (epoch 185 vs 105 for minimal)

### 4. GAT - Catastrophic Test Set Failure

**Minimal Trainer:**
- Validation: 0.1190 (epoch 65)
- Test: 0.0728
- Val-Test Gap: 0.0462 (38.9%)
- Training: Early convergence with early stopping at epoch 165

**Morgan Features:**
- Validation: 0.0432 (epoch 180)
- Test: 0.0112
- Val-Test Gap: 0.0320 (74.0%)
- **Performance Drop: 63.7% on validation, 84.6% on test**

**Analysis:**
- **Most catastrophic failure on test set** (84.6% drop)
- Large val-test gap indicates severe overfitting
- Attention heads may be focusing on spurious patterns in Morgan features
- Multi-head attention (8 heads) may be over-parameterized for this feature space
- The model appears to memorize validation patterns without generalizing

---

## Root Cause Analysis

### Why Did Morgan Features Hurt Performance?

1. **Feature Dimensionality Mismatch**
   - Morgan fingerprints: 2048 dimensions
   - Graph has only 4,267 nodes
   - High ratio may lead to overfitting
   - Minimal baselines relied purely on graph structure (more robust)

2. **Information Redundancy**
   - Drug-drug interactions may be better captured by graph topology
   - Morgan features capture molecular structure, not interaction patterns
   - DDI prediction may be more about graph connectivity than chemical similarity
   - Features may introduce noise rather than signal

3. **Training Dynamics Issues**
   - All Morgan models took longer to converge
   - Higher dimensional gradients may complicate optimization
   - No feature normalization was explicitly mentioned (potential issue)
   - Learning rate may not be optimal for feature-based training

4. **Architectural Limitations**
   - Current architectures may not be designed to leverage dense chemical features
   - Simple aggregation (sum/mean) may dilute informative features
   - No gating or feature selection mechanisms
   - Fixed architecture may be too simple for feature-rich inputs

5. **Data Split Sensitivity**
   - Morgan features may capture validation-specific patterns
   - Test set distribution may differ from validation
   - Graph structure alone may generalize better across splits

---

## Generalization Analysis

### Val-Test Gap Comparison

| Model | Minimal Gap | Morgan Gap | Change |
|-------|------------|------------|---------|
| GCN | 11.0% | 12.0% | +1.0% |
| GraphSAGE | 54.1% | -54.2% (inverted) | **-108.3%** |
| GraphTransformer | 54.0% | 3.7% | **-50.3%** (better) |
| GAT | 38.9% | 74.0% | **+35.1%** (worse) |

**Key Findings:**
- **GraphSAGE:** Completely inverted behavior (test > val)
- **GraphTransformer:** Dramatically improved gap (potential underfitting)
- **GAT:** Much worse generalization with Morgan features
- **GCN:** Most stable across both setups

---

## Training Efficiency

### Convergence Speed

| Model | Minimal Best Epoch | Morgan Best Epoch | Training Time Impact |
|-------|------------------:|------------------:|---------------------|
| GCN | 180 | 185 | Similar |
| GraphSAGE | 200 | 200 | Similar (but worse) |
| GraphTransformer | 105 | 185 | **+76% epochs needed** |
| GAT | 65 (early stop) | 180 | **+177% epochs needed** |

**Observations:**
- Morgan features significantly slowed convergence for attention-based models
- GraphTransformer and GAT needed 2-3x more epochs
- Suggests optimization landscape is more complex with features
- May benefit from different learning rates or schedules

---

## Recommendations

### Immediate Actions

1. **Investigate Feature Preprocessing**
   - Normalize Morgan fingerprints (z-score or min-max)
   - Apply PCA or feature selection to reduce dimensionality
   - Try different Morgan fingerprint radii (currently unknown)

2. **Architectural Modifications**
   - Add feature projection layers before graph convolutions
   - Implement feature gating mechanisms
   - Try lower hidden dimensions to match feature scale
   - Add batch normalization after feature transformations

3. **Hybrid Approaches**
   - Combine graph structure and features more explicitly
   - Try feature-only baselines (MLP) for comparison
   - Implement residual connections that allow bypassing features

4. **Alternative Features**
   - Try different chemical descriptors (ECFP4, MACCS keys)
   - Use learned embeddings instead of fixed fingerprints
   - Consider protein target features instead of/in addition to chemical features

### Long-term Strategy

1. **Feature Engineering**
   - Investigate which Morgan bits are most informative
   - Create interaction-specific features (e.g., shared substructures)
   - Incorporate pharmacological properties (ATC codes, targets)

2. **Model Architecture**
   - Design architectures specifically for feature-rich graphs
   - Implement attention over features, not just topology
   - Try graph transformers with feature-aware positional encodings

3. **Training Procedures**
   - Hyperparameter sweep specifically for Morgan models
   - Try different optimizers (AdamW, LAMB)
   - Implement curriculum learning (structure → features)

4. **Evaluation**
   - Analyze which types of edges benefit from features
   - Compare performance on different drug classes
   - Investigate if rare interactions benefit more from features

---

## Conclusions

1. **Graph structure alone outperforms chemical features** for DDI prediction on ogbl-ddi
2. **Morgan fingerprints introduce more noise than signal** with current architectures
3. **Attention-based models (GAT, Transformer) are most negatively impacted** by features
4. **GCN shows relative robustness** but still significant degradation
5. **The current feature integration approach is fundamentally flawed** and requires redesign

**The minimal trainer approach should be considered the baseline to beat.** Adding features should only be pursued with significant architectural changes and careful feature engineering.

---

## FAQ: Would Increasing Model Size/Complexity Help?

**Short answer: No, likely not.**

### Current Architecture Analysis

**Parameter counts:**
- Projection: 2048 → 128 = ~262K parameters
- GNN layers: 2 × 128 hidden = ~33K parameters  
- Total: ~295K parameters for only 4,267 nodes
- **Ratio: 69 parameters per node**

### Why More Complexity Would Hurt

1. **Empirical Evidence**: Complexity already correlates with worse performance
   - GCN (simplest): -46% degradation
   - GAT (most complex): -85% test degradation
   - More parameters → more overfitting on noisy features

2. **Dataset Size**: Only 4,267 nodes, already parameter-rich
   - More parameters would increase overfitting
   - Need ~10-20x more data to justify larger models

3. **Signal vs Capacity**: Minimal trainer with NO features performed better
   - Problem is feature quality, not model capacity
   - Bigger model will just memorize noise better

4. **Bottleneck Analysis**: 2048→128 projection already aggressive
   - Increasing to 512 dims would create 1M+ params in first layer
   - Would preserve more noise, not more signal

### What Would Actually Help

**Instead of scaling up, try:**

1. **Feature Selection/Reduction**
   ```python
   # PCA to 64-128 dims instead of all 2048
   # Feature selection based on variance/correlation
   # Sparse fingerprints (keep only active bits)
   ```

2. **Better Feature Integration**
   ```python
   # Multi-layer projection: 2048 → 512 → 256 → 128
   # Gating: learn to mix features with structure
   # Residual: allow bypassing features entirely
   ```

3. **Regularization (not expansion)**
   ```python
   # L1/L2 on projection layer to kill noisy features
   # Feature dropout (randomly zero Morgan bits)
   # Batch normalization on features
   ```

4. **Architectural Changes**
   ```python
   # Separate feature and structure encoders
   # Attention over features (which bits to use)
   # Learnable feature embeddings instead of fixed
   ```

### The Core Insight

**Graph structure alone >> Graph structure + Morgan features**

This tells us the features are fundamentally problematic for this task. Scaling up won't fix bad inputs—it will only help the model memorize validation patterns that don't generalize.

**Recommended approach:** Fix the features first (selection, normalization, alternative descriptors), then consider scaling if performance plateaus.

---

## Appendix: Full Training Logs

**Minimal Trainer Log:** `logs/baselines_20251209_174601/baselines.log`  
**Morgan Baselines Log:** `logs/morgan_baselines_20251209_191058/morgan_baselines.log`


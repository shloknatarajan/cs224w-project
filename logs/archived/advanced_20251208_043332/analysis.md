# GCN Structural V3 - Analysis Report

**Date:** 2025-12-08
**Log File:** `logs/advanced_20251208_043332/advanced.log`
**Model:** GCN-Structural-V3

---

## 1. Changes from V2 to V3

### Architecture Changes

| Feature | V2 | V3 | Impact |
|---------|----|----|--------|
| **Layers** | 2 | 3 | More capacity for learning |
| **Activation** | ReLU | LeakyReLU (α=0.2) | Prevents dead neurons |
| **Residual Connections** | Only after layer 1 | **ALL layers** | Better gradient flow |
| **Residual Scaling** | 0.3 | 0.5 | Stronger skip connections |
| **Diversity Loss Weight** | 0.01 | 0.05 (5x stronger) | Fights embedding collapse more aggressively |

### Key Code Differences

**V2 Residual Pattern (line 96-97):**
```python
if i > 0:
    x = x + 0.3 * x_prev  # Only for layers after first
```

**V3 Residual Pattern (line 100-102):**
```python
# Residual connection on ALL layers (including first layer)
# Scale factor helps with gradient flow
x = x + 0.5 * x_prev  # Always applied
```

**V2 Activation (line 93):**
```python
x = F.relu(x)
```

**V3 Activation (line 97):**
```python
x = F.leaky_relu(x, negative_slope=0.2)
```

### Training Configuration
- **Epochs:** 400 with patience 20
- **Learning Rate:** 0.005 → 0.0025 (LR decay at epoch 125)
- **Hard Negative Sampling:** Activated at epoch 55 (ratio=0.05)
- **Batch Size:** 20,000 (gradient accumulation: 3 steps)
- **Decoder:** Simple (not multi-strategy)

---

## 2. Results Analysis

### Performance Summary

| Metric | Best Epoch | Validation Hits@20 | Test Hits@20 |
|--------|------------|-------------------|--------------|
| **Peak Performance** | 160 | **0.0022** | **0.0028** |
| First improvement | 40 | 0.0002 | 0.0003 |
| Secondary peak | 145 | 0.0022 | 0.0030 |

**Status:** ⚠️ **CRITICALLY POOR PERFORMANCE** - The model is essentially failing to predict any links (0.22% validation accuracy).

### Embedding Collapse Analysis

The model showed **progressive improvement** in embedding diversity:

| Epoch | sim_mean | sim_std | Status |
|-------|----------|---------|--------|
| 1 | 0.994 | 0.026 | ❌ Severe collapse |
| 10 | 0.993 | 0.029 | ❌ Severe collapse |
| 20 | 0.969 | 0.089 | ❌ Collapse |
| 50 | 0.836 | 0.222 | ⚠️ Improving |
| 100 | 0.682 | 0.244 | ✓ Better |
| 130 | 0.609 | 0.213 | ✓ Good |
| 160 | 0.568 | 0.210 | ✓ Good |

**Key Finding:** The stronger diversity loss (0.05) successfully reduced embedding similarity from 0.994 → 0.568 over 160 epochs. However, **diversity alone is not sufficient** for good link prediction performance.

### Score Gap Analysis

The gap between positive and negative edge scores improved:

| Epoch | pos_score | neg_score | gap | Link Prediction Quality |
|-------|-----------|-----------|-----|------------------------|
| 1 | 0.983 | 0.983 | 0.001 | ❌ No discrimination |
| 50 | 0.671 | 0.322 | 0.349 | ⚠️ Some separation |
| 100 | 0.842 | 0.298 | 0.544 | ✓ Good separation |
| 160 | 0.818 | 0.203 | 0.615 | ✓ Excellent separation |

**Paradox:** Despite excellent score separation (gap=0.615), validation Hits@20 remains at 0.0022. This suggests a **calibration problem** or **distribution mismatch** between training and validation data.

### Gradient Flow

Gradient norms were healthy throughout training:
- Epoch 1: 10.01 (strong gradients)
- Epoch 50-160: 0.13-2.53 (stable range)

**Diagnostic Issue Resolution:** The earlier V2 diagnostic showed gradient norm = 0.00 due to checking **after** `optimizer.zero_grad()`. This was a measurement bug, not a real gradient vanishing issue.

---

## 3. Problem Diagnosis

### Why is Performance So Poor?

Despite all improvements (diversity ✓, score separation ✓, gradients ✓), the model achieves only **0.22% validation accuracy**. Possible causes:

1. **Hard Negative Sampling Too Aggressive**
   - Started at epoch 55 with ratio=0.05
   - May be sampling negatives that are too hard, causing the model to overfit to training distribution
   - Training loss decreased (1.87 → 0.72) but validation didn't improve

2. **Validation/Test Distribution Mismatch**
   - Model learns to separate training edges well (gap=0.615)
   - But this doesn't transfer to validation/test edges
   - Possible temporal or structural difference in edge splits

3. **Decoder Limitations**
   - Using simple dot product decoder
   - Multi-strategy decoder was disabled (`use_multi_strategy=False`)
   - May need more expressive decoder (MLP, DistMult, etc.)

4. **Over-regularization**
   - Diversity loss (0.05) + Edge dropout (0.15) + Feature dropout (0.2)
   - Model may be too constrained to learn meaningful patterns

5. **Insufficient Capacity**
   - ogbl-ddi has 1M+ edges, but only 4,267 nodes
   - High edge density → complex interaction patterns
   - 3 layers with 256 hidden dim may not be enough

---

## 4. Next Steps

### Immediate Experiments

#### A. Decoder Exploration
**Priority: HIGH** - Most likely bottleneck

1. **Enable Multi-Strategy Decoder**
   ```python
   use_multi_strategy=True  # Currently False
   ```
   - Combines multiple edge scoring strategies
   - Reference: V2 analysis showed multi-strategy helped

2. **Try MLP Decoder**
   - Add 2-layer MLP: `[hidden_dim*2 → hidden_dim → 1]`
   - More expressive than dot product

3. **Try DistMult Decoder**
   - Learn a relation-specific weight matrix
   - Better for dense graphs

#### B. Hard Negative Tuning
**Priority: HIGH** - May be causing overfitting

1. **Reduce Hard Negative Ratio**
   - Try `hard_neg_ratio=0.01` (currently 0.05)
   - Start hard negatives later (epoch 100 instead of 55)

2. **Disable Hard Negatives**
   - Baseline experiment: `use_hard_negatives=False`
   - Check if validation improves

#### C. Regularization Adjustment
**Priority: MEDIUM**

1. **Reduce Diversity Loss Weight**
   - Try `diversity_weight=0.02` or `0.01` (currently 0.05)
   - We achieved good diversity, may not need it as strong

2. **Reduce Dropout**
   - Try `dropout=0.1` (currently 0.2)
   - Edge dropout already provides regularization

#### D. Model Capacity
**Priority: MEDIUM**

1. **Increase Hidden Dimension**
   - Try `hidden_dim=512` (currently 256)
   - More capacity for complex patterns

2. **Add More Layers**
   - Try `num_layers=4` (currently 3)
   - But watch for over-smoothing

#### E. Learning Strategy
**Priority: LOW**

1. **Different LR Schedule**
   - Try cosine annealing instead of step decay
   - Warmup phase: LR 0.001 → 0.005 over first 50 epochs

2. **Lower Initial LR**
   - Try `lr=0.001` (currently 0.005)
   - More stable training

---

## 5. Recommended Action Plan

### Phase 1: Decoder Experiments (1-2 runs)
1. Run V3 with `use_multi_strategy=True`
2. Compare performance vs current V3

### Phase 2: Hard Negative Tuning (2-3 runs)
1. Disable hard negatives entirely
2. If improves: tune ratio (0.01, 0.02) and start epoch (100, 150)

### Phase 3: Combined Best Approach (1 run)
1. Best decoder + Best hard negative config
2. Consider reducing diversity weight to 0.02

### Phase 4: Capacity Increase (if needed)
1. If still poor: try hidden_dim=512
2. If still poor: investigate dataset splits and preprocessing

---

## 6. Key Insights

✅ **What Worked:**
- LeakyReLU prevented dead neurons
- Full residual connections maintained gradient flow
- Stronger diversity loss successfully reduced embedding collapse
- Score separation improved significantly (gap: 0.001 → 0.615)

❌ **What Didn't Work:**
- Despite architectural improvements, validation performance remains critically poor
- The model learned *something* (good score separation) but not the *right thing* (link prediction)

🔍 **Core Issue:**
The problem has shifted from **gradient/collapse issues** to **representation/calibration issues**. The model's internal representations don't align with what's needed for validation performance.

---

## 7. Code References

- V3 Model: `src/models/advanced/gcn_structural_v3.py`
- V2 Model: `src/models/advanced/gcn_structural_v2.py`
- Training Log: `logs/advanced_20251208_043332/advanced.log`

---

**Conclusion:** GCN-V3 successfully addressed gradient flow and embedding collapse, but link prediction performance remains unacceptably low. The next priority is exploring different decoders and adjusting hard negative sampling strategy.

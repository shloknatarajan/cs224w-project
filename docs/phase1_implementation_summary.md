# Phase 1 Implementation Summary
**Date:** 2025-12-08
**Status:** ✅ Implemented and Running

## Overview

Successfully implemented Phase 1 improvements from `baseline_improvement_plan_20251207.md` to address the two most critical issues in link prediction performance.

## Changes Implemented

### 1. Hard Negative Sampling (CRITICAL) ✅

**File:** `src/training/losses.py`

**New Functions:**
- `k_hop_negative_sampling()`: Samples negatives from k-hop neighborhoods
  - More informative than random sampling
  - Samples from "close but not connected" nodes
  - Configurable k (2-3 recommended)
  - Falls back to random sampling if insufficient k-hop neighbors

**Enhancement:** `src/training/trainer.py`
- Added `neg_sampling_strategy` parameter with 4 options:
  - `'random'`: Random negative sampling (baseline)
  - `'hard'`: Score-based hard negative mining (existing)
  - `'khop'`: K-hop neighborhood sampling (NEW)
  - `'mixed'`: 50/50 mix of score-based and k-hop (NEW)
- Added `k_hop` parameter to control hop distance (default: 2)

**Training Impact:**
- Enabled hard negatives for all models (`use_hard_negatives=True`)
- Using score-based hard negative strategy by default
- 30% hard negatives + 70% random negatives after 50 warmup epochs

### 2. MLP Decoder (CRITICAL) ✅

**File:** `src/models/base.py`

**Existing Infrastructure (Activated):**
- `ImprovedEdgeDecoder` with `use_multi_strategy=True`
- **Multi-strategy decoder** combines three approaches:
  1. **Hadamard path**: `h_u ⊙ h_v` → MLP(hidden_dim//2) → output
  2. **Concatenation path**: `[h_u || h_v]` → MLP(hidden_dim) → output
  3. **Bilinear path**: Bilinear(h_u, h_v) → output
- Learnable weights combine strategies with softmax normalization
- Significantly more expressive than simple dot product

**Training Impact:**
- All models now use `use_multi_strategy=True`
- Replaces simple dot product decoder from baselines

### 3. Per-Model Learning Rate Tuning ✅

**File:** `train_phase1.py`

**Learning Rate Adjustments:**
- **GCN**: 0.015 (↑ from 0.01) - Leverage strong baseline performance
- **GraphSAGE**: 0.003 (↓ from 0.01) - Fix underfitting issue
- **GraphTransformer**: 0.005 (unchanged) - Already well-tuned
- **GAT**: 0.005 (unchanged) - Already well-tuned

**Rationale:**
- GCN showed best performance (0.57%) → increase LR to converge faster
- GraphSAGE worst performer (0.15%) + signs of underfitting → reduce LR
- Transformer/GAT already using 0.005 which was working well

## Files Modified

1. **`src/training/losses.py`**
   - Added `k_hop_negative_sampling()` function

2. **`src/training/trainer.py`**
   - Imported `k_hop_negative_sampling`
   - Added `neg_sampling_strategy` and `k_hop` parameters
   - Updated negative sampling logic to support 4 strategies
   - Enhanced logging to show sampling strategy

3. **`train_phase1.py`** (NEW)
   - Created Phase 1 training script
   - Enables hard negatives + multi-strategy decoder
   - Implements per-model LR tuning
   - Includes improvement analysis vs baseline

## Configuration Details

### Common Hyperparameters (Unchanged)
```python
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5
DECODER_DROPOUT = 0.3
EPOCHS = 200
PATIENCE = 20
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 50000
GRADIENT_ACCUMULATION_STEPS = 1
```

### Phase 1 Specific Settings
```python
USE_HARD_NEGATIVES = True
USE_MULTI_STRATEGY = True  # MLP decoder
HARD_NEG_RATIO = 0.3  # 30% hard, 70% random
NEG_SAMPLING_STRATEGY = 'hard'  # Score-based
```

## Expected Improvements

Based on improvement plan projections:

| Model | Baseline Val | Expected Val | Expected Improvement |
|-------|--------------|--------------|---------------------|
| **GCN** | 0.57% | 1.7-3.4% | 3-6x |
| **GraphSAGE** | 0.15% | 0.6-1.5% | 4-10x |
| **GraphTransformer** | 0.29% | 0.8-1.7% | 3-6x |
| **GAT** | 0.25% | 0.75-1.5% | 3-6x |

**Key Metrics to Watch:**
- Validation Hits@20 improvement
- Val-Test gap reduction (overfitting)
- Training stability and convergence

## Running the Experiments

```bash
# Start Phase 1 training (all 4 models)
pixi run python train_phase1.py

# Monitor progress
tail -f logs/phase1_*.log

# Check for latest log
ls -lth logs/phase1_*.log | head -1
```

## Next Steps (Phase 2)

After Phase 1 results:
1. Analyze which models benefited most
2. Test k-hop negative sampling (`neg_sampling_strategy='khop'`)
3. Implement model-specific regularization tuning
4. Try mixed negative sampling strategy

## Implementation Notes

### Advantages of Current Approach
- ✅ Minimal code changes (leveraged existing infrastructure)
- ✅ Backward compatible (baseline script unchanged)
- ✅ Easy to experiment with different strategies
- ✅ Comprehensive logging for analysis

### K-hop Sampling Implementation Details
- Uses BFS to find k-hop neighbors
- Excludes 1-hop neighbors (those are positive edges)
- Verifies sampled edges aren't in the graph
- Falls back to random if insufficient k-hop neighbors
- May be slower than score-based sampling (builds adjacency lists)

### Future Optimizations
- Pre-compute k-hop neighborhoods for efficiency
- Cache adjacency structures between epochs
- Parallelize k-hop computation
- Add distance-based weighting for k-hop sampling

## Monitoring Checklist

During training, watch for:
- [ ] Hard negatives activate after epoch 50
- [ ] Loss convergence patterns
- [ ] Validation improvement vs baseline
- [ ] Val-Test gap reduction
- [ ] Learning rate scheduler adjustments
- [ ] Memory usage (should be similar to baseline)
- [ ] Training time per epoch (~7-8 seconds expected)

## Logging Format

Phase 1 logs include:
- Configuration summary at start
- Negative sampling strategy used
- Multi-strategy decoder confirmation
- Per-epoch metrics with improvement markers (🔥)
- Final comparison with baseline results
- Improvement percentages

Example log entry:
```
GCN-Phase1 Epoch 0100/200 [Hard Neg] | Loss: 0.8234 | Val Hits@20: 0.0152 | Test Hits@20: 0.0098 | Best Val: 0.0152 (epoch 100) | LR: 0.015000 🔥
```

# Hyperparameter Search Experiments

## Overview

This document outlines the hyperparameter search strategy for minimal baseline models (GCN, GraphSAGE, GAT, GraphTransformer) on the OGBL-DDI dataset.

## Motivation

The current baseline models use ultra-minimal configurations (dropout=0, simple decoders) that achieved significant improvements over the original broken configurations. However, these may not be optimal. This hyperparameter search aims to:

1. **Find optimal configurations** for each model architecture
2. **Understand sensitivity** to key hyperparameters
3. **Identify patterns** across different architectures
4. **Balance performance vs. generalization** (val-test gap)

## Search Strategy

### Three-Tier Approach

#### 1. Minimal Search (Recommended for quick exploration)
- **Time**: ~2-4 hours per model
- **Configurations**: 36 per model (GCN), 72 (GraphSAGE), 72 (GAT/Transformer)
- **Purpose**: Identify promising parameter ranges quickly

**Search Space:**
```python
{
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3],
    'dropout': [0.0, 0.1, 0.2],
    'decoder_dropout': [0.0],
    'use_multi_strategy': [False],
    'heads': [2, 4],  # GAT/Transformer only
    'use_batch_norm': [False, True],  # GraphSAGE only
    'lr': [0.01, 0.005],
    'weight_decay': [1e-4, 1e-3],
    'batch_size': [50000],
}
```

#### 2. Standard Search (Balanced)
- **Time**: ~1-2 days per model
- **Configurations**: 288+ per model
- **Purpose**: Thorough exploration with multiple decoder strategies

**Search Space:**
```python
{
    'hidden_dim': [64, 128, 256, 512],
    'num_layers': [2, 3, 4],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'decoder_dropout': [0.0, 0.1],
    'use_multi_strategy': [False, True],
    'heads': [2, 4, 8],
    'use_batch_norm': [False, True],
    'lr': [0.001, 0.005, 0.01, 0.02],
    'weight_decay': [1e-5, 1e-4, 1e-3],
    'batch_size': [25000, 50000, 100000],
}
```

#### 3. Extensive Search (Comprehensive)
- **Time**: ~3-5 days per model
- **Configurations**: 1000+ per model
- **Purpose**: Find absolute best configuration

**Search Space:**
```python
{
    'hidden_dim': [32, 64, 128, 256, 512],
    'num_layers': [2, 3, 4, 5],
    'dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    'decoder_dropout': [0.0, 0.05, 0.1, 0.2],
    'use_multi_strategy': [False, True],
    'heads': [2, 4, 8, 16],
    'use_batch_norm': [False, True],
    'lr': [0.0005, 0.001, 0.005, 0.01, 0.02],
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
    'batch_size': [25000, 50000, 100000],
}
```

## Key Hyperparameters

### 1. Architecture Parameters

#### Hidden Dimension (`hidden_dim`)
- **Impact**: High - Affects model capacity
- **Current**: 128
- **Range to explore**: 64, 128, 256, 512
- **Hypothesis**: Larger dimensions may help for dense graphs like DDI

#### Number of Layers (`num_layers`)
- **Impact**: High - Controls receptive field
- **Current**: 2
- **Range to explore**: 2, 3, 4, 5
- **Hypothesis**: More layers may capture longer-range dependencies
- **Risk**: Over-smoothing on dense graphs

#### Dropout (`dropout`)
- **Impact**: High - Critical for regularization
- **Current**: 0.0 (proven to work)
- **Range to explore**: 0.0, 0.05, 0.1, 0.15, 0.2, 0.3
- **Hypothesis**: Some dropout may help generalization without hurting too much
- **Known issue**: High dropout (>0.3) destroyed performance in original configs

#### Decoder Dropout (`decoder_dropout`)
- **Impact**: Medium - Regularizes link prediction head
- **Current**: 0.0
- **Range to explore**: 0.0, 0.05, 0.1, 0.2
- **Hypothesis**: May help reduce val-test gap

### 2. Decoder Strategy

#### Multi-Strategy Decoder (`use_multi_strategy`)
- **Impact**: Medium-High - Combines multiple edge embeddings
- **Current**: False (simple Hadamard product)
- **Options**: False (simple), True (multi-strategy)
- **Hypothesis**: Multi-strategy may capture more complex interaction patterns

### 3. Model-Specific Parameters

#### Attention Heads (`heads` - GAT/Transformer)
- **Impact**: Medium - Affects expressiveness of attention
- **Current**: 2
- **Range to explore**: 2, 4, 8, 16
- **Hypothesis**: More heads may help but increase memory

#### Batch Normalization (`use_batch_norm` - GraphSAGE)
- **Impact**: Medium - Normalizes activations
- **Current**: False
- **Options**: False, True
- **Hypothesis**: May stabilize training and improve generalization

### 4. Training Parameters

#### Learning Rate (`lr`)
- **Impact**: High - Critical for convergence
- **Current**: 0.01 (GCN, SAGE), 0.005 (GAT, Transformer)
- **Range to explore**: 0.0005, 0.001, 0.005, 0.01, 0.02
- **Hypothesis**: Optimal LR varies by architecture

#### Weight Decay (`weight_decay`)
- **Impact**: Medium - L2 regularization
- **Current**: 1e-4
- **Range to explore**: 0.0, 1e-5, 1e-4, 1e-3
- **Hypothesis**: Higher weight decay may reduce val-test gap

#### Batch Size (`batch_size`)
- **Impact**: Medium - Affects gradient noise and memory
- **Current**: 50000
- **Range to explore**: 25000, 50000, 100000
- **Hypothesis**: Larger batches may be more stable for dense graphs

## Experimental Protocol

### 1. Setup
```bash
# Run minimal search on GCN first
python tune_baselines.py
```

Edit `tune_baselines.py` to change:
- `SEARCH_TYPE`: 'minimal', 'standard', or 'extensive'
- `MODELS_TO_SEARCH`: ['GCN', 'GraphSAGE', 'GAT', 'GraphTransformer']

### 2. Execution Order

**Phase 1: Quick Validation (Minimal Search)**
1. GCN - Establish baseline improvements
2. GraphSAGE - Test on best-performing architecture
3. GAT - Explore attention mechanisms
4. GraphTransformer - Compare with GAT

**Phase 2: Deep Dive (Standard Search)**
- Focus on top 2 architectures from Phase 1
- Explore multi-strategy decoders
- Test batch size variations

**Phase 3: Final Optimization (Extensive Search)**
- Run only on the best model from Phase 2
- Fine-grained parameter sweeps

### 3. Evaluation Metrics

For each configuration, track:
- **Best Validation Hits@20**: Primary optimization target
- **Best Test Hits@20**: True performance measure
- **Val-Test Gap**: `val_hits - test_hits` (measures generalization)
- **Best Epoch**: Early stopping point (training efficiency)

**Good Configuration Criteria:**
1. High test Hits@20 (>15% for baselines)
2. Small val-test gap (<3% absolute)
3. Reasonable training time (<50 epochs to best val)

## Analysis Plan

### 1. Statistical Analysis

After running experiments, analyze:

#### Parameter Importance
- Which parameters have the strongest correlation with test performance?
- Which combinations work well together?

#### Generalization Patterns
- Which settings minimize val-test gap?
- Is there a trade-off between peak performance and generalization?

#### Architecture Comparison
- How do optimal configs differ across architectures?
- Which architecture is most robust to hyperparameter choices?

### 2. Visualization

Create plots for:
1. **Performance Distribution**: Box plots per hyperparameter value
2. **Parameter Correlation**: Heatmap of parameter combinations
3. **Pareto Frontier**: Test performance vs. val-test gap
4. **Training Efficiency**: Test performance vs. epochs to convergence

### 3. Recommendations

Generate:
1. **Best Overall Config**: Highest test Hits@20
2. **Best Generalizing Config**: Smallest val-test gap with good performance
3. **Best Efficient Config**: Good performance with fastest convergence
4. **Model-Specific Recommendations**: Top config per architecture

## Expected Outcomes

### Performance Targets

Based on current baseline results:
- **GCN**: Currently 13-24% → Target: >25%
- **GraphSAGE**: Currently 17% → Target: >20% (best baseline)
- **GAT**: Currently 11% → Target: >15%
- **GraphTransformer**: Currently 8% → Target: >12%

### Key Questions to Answer

1. **Is dropout beneficial?** Current configs use 0, but small dropout may help
2. **Do multi-strategy decoders help?** Trade-off between complexity and performance
3. **What's the optimal architecture size?** Balance between capacity and overfitting
4. **Which model generalizes best?** Minimize val-test gap
5. **Are attention mechanisms worth it?** GAT/Transformer vs. GCN/SAGE

## Running the Experiments

### Quick Start

```bash
# Minimal search on GCN only (fastest)
python tune_baselines.py

# Modify the script to search all models:
# Change MODELS_TO_SEARCH = ['GCN', 'GraphSAGE', 'GAT', 'GraphTransformer']
```

### Parallel Execution

For faster results, run models in parallel (if you have multiple GPUs):

```bash
# Terminal 1 - GCN
python tune_baselines.py  # Set MODELS_TO_SEARCH = ['GCN']

# Terminal 2 - GraphSAGE
python tune_baselines.py  # Set MODELS_TO_SEARCH = ['GraphSAGE']

# Terminal 3 - GAT
python tune_baselines.py  # Set MODELS_TO_SEARCH = ['GAT']

# Terminal 4 - GraphTransformer
python tune_baselines.py  # Set MODELS_TO_SEARCH = ['GraphTransformer']
```

### Monitor Progress

Results are logged in real-time to:
- `logs/hyperparam_search_TIMESTAMP/search.log`
- `logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json`

### Analyze Results

After completion:
```bash
# View summary
tail -n 100 logs/hyperparam_search_TIMESTAMP/search.log

# Analyze JSON results
python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json
```

## Tips for Success

1. **Start small**: Run minimal search first to validate the setup
2. **Monitor early**: Check first few configs to ensure training is working
3. **Watch for OOM**: If CUDA out of memory, reduce batch_size or hidden_dim
4. **Compare fairly**: All models use same train/val/test splits and evaluation
5. **Document findings**: Note any anomalies or interesting patterns
6. **Iterate**: Use insights from minimal search to refine standard search space

## Next Steps

After finding optimal configurations:

1. **Re-run best configs** 3-5 times to measure variance
2. **Update baseline scripts** with new optimal settings
3. **Compare with SEAL** and other advanced methods
4. **Write up findings** in paper/report
5. **Consider ensemble methods** combining top models

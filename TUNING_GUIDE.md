# Hyperparameter Tuning Guide for Hybrid GCN

This guide explains how to use the hyperparameter tuning scripts to improve the Hybrid GCN model.

## Current Baseline

**Model:** Hybrid-GCN-Simple
**Performance:** val@20=0.2349, test@20=0.1933
**Configuration:**
```python
{
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.0,
    'decoder_dropout': 0.3,
    'lr': 0.01,
    'weight_decay': 1e-4,
    'batch_size': 5000,
}
```

## Available Tuning Scripts

### 1. `tune_hybrid_gcn_quick.py` - Quick Exploration (Recommended First)

**Use case:** Fast initial exploration to find promising configurations
**Runtime:** ~20 experiments × 10-15 min = 3-5 hours
**Settings:** 50 epochs, patience=10

**Hyperparameters explored:**
- `hidden_dim`: [96, 128, 160, 256]
- `num_layers`: [2, 3]
- `decoder_dropout`: [0.2, 0.3, 0.4]
- `lr`: [0.005, 0.01, 0.015, 0.02]
- `weight_decay`: [5e-5, 1e-4, 2e-4, 5e-4]

**How to run:**
```bash
python tune_hybrid_gcn_quick.py
```

**When to use:**
- Initial exploration to get a sense of the parameter space
- Testing hypotheses about which parameters matter most
- When you want results quickly

### 2. `tune_hybrid_gcn.py` - Thorough Search

**Use case:** Comprehensive search with full training
**Runtime:** ~30 experiments × 40-60 min = 20-30 hours
**Settings:** 200 epochs, patience=20 (same as baseline)

**Hyperparameters explored:**
- `hidden_dim`: [64, 128, 256]
- `num_layers`: [2, 3, 4]
- `dropout`: [0.0, 0.1]
- `decoder_dropout`: [0.2, 0.3, 0.4, 0.5]
- `lr`: [0.005, 0.01, 0.02]
- `weight_decay`: [1e-5, 1e-4, 5e-4, 1e-3]
- `batch_size`: [5000, 10000, 20000]

**Search strategies:**
1. **Random search** (default): Randomly samples configurations
2. **Focused search**: Starts with baseline, then varies one parameter at a time

**How to run:**
```bash
# Random search (default)
python tune_hybrid_gcn.py

# To switch to focused search, edit line ~370 in the script:
# SEARCH_STRATEGY = "focused"
```

**When to use:**
- After quick exploration has identified promising regions
- When you want to find the absolute best configuration
- For final model selection

## Recommended Workflow

### Step 1: Quick Exploration
```bash
python tune_hybrid_gcn_quick.py
```

**What this does:**
- Tests 20 random configurations quickly
- Identifies promising parameter ranges
- Takes 3-5 hours

**Check the results:**
```bash
# View top configurations
tail -100 logs/hybrid_gcn_quick_tuning_*/quick_tuning.log

# Or check the JSON results
cat logs/hybrid_gcn_quick_tuning_*/results.json | jq '.[] | select(.success==true) | {val_hits, config}' | less
```

### Step 2: Focused Refinement

Based on quick results, either:

**Option A:** Run full training on top 3-5 configs from quick search
```bash
# Manually create a script with the best configs and run them with 200 epochs
# Or modify tune_hybrid_gcn.py to only test those specific configs
```

**Option B:** Run thorough random search
```bash
python tune_hybrid_gcn.py
```

### Step 3: Analysis

After experiments complete, check the results:

```bash
# View summary
tail -200 logs/hybrid_gcn_tuning_*/tuning.log

# View detailed results
cat logs/hybrid_gcn_tuning_*/results.json | jq '.[] | select(.success==true) | {exp_id, val_hits, improvement_pct, config}' | less

# Find best configuration
cat logs/hybrid_gcn_tuning_*/results.json | jq '[.[] | select(.success==true)] | sort_by(-.val_hits) | .[0]'
```

## Understanding the Results

### Result Files

Each experiment creates a log directory with:
- `tuning.log` or `quick_tuning.log`: Full training logs
- `results.json`: Structured results for all experiments

### Key Metrics

- **val@20**: Validation Hits@20 score (higher is better)
- **test@20**: Test Hits@20 score (for best validation epoch)
- **improvement_pct**: Percentage improvement over baseline
- **best_epoch**: Epoch with best validation score

### Interpreting Results

The logs include:

1. **Top 10 configurations**: Best performing configs ranked by val@20
2. **Parameter impact analysis**: Shows average performance for each parameter value
3. **Summary statistics**: Overall improvement statistics

Look for:
- Configs with improvement > +5% (significant improvement)
- Consistent patterns (e.g., higher hidden_dim always better)
- Parameter interactions (e.g., higher lr works with higher weight_decay)

## Hyperparameter Descriptions

### Model Architecture

- **hidden_dim**: Size of node embeddings (64, 128, 256)
  - Larger → more capacity, slower training
  - Current baseline: 128

- **num_layers**: Number of GCN layers (2, 3, 4)
  - More layers → larger receptive field, risk of oversmoothing
  - Current baseline: 2

- **dropout**: Dropout rate in GCN layers (0.0, 0.1)
  - Higher → more regularization, prevents overfitting
  - Current baseline: 0.0 (no dropout in encoder)

- **decoder_dropout**: Dropout in decoder MLPs (0.2-0.5)
  - Helps prevent overfitting in decoder
  - Current baseline: 0.3

### Training Parameters

- **lr**: Learning rate (0.005-0.02)
  - Higher → faster learning, risk of instability
  - Current baseline: 0.01

- **weight_decay**: L2 regularization (1e-5 to 1e-3)
  - Higher → stronger regularization
  - Current baseline: 1e-4

- **batch_size**: Edges per training batch (5000-20000)
  - Larger → more stable gradients, higher memory usage
  - Current baseline: 5000

## Tips for Effective Tuning

### 1. Start Small
Use `tune_hybrid_gcn_quick.py` first to understand the parameter space.

### 2. Monitor GPU Memory
If you get OOM errors:
- Reduce `batch_size`
- Reduce `hidden_dim`
- Add `torch.cuda.empty_cache()` between experiments

### 3. Save Promising Models
The scripts currently don't save models. To save the best model:
```python
# Add after train_hybrid_model() call
if result.best_val_hits > best_so_far:
    torch.save(model.state_dict(), f'best_model_exp{exp_id}.pt')
```

### 4. Parallel Experiments
To run multiple searches in parallel (if you have multiple GPUs):
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python tune_hybrid_gcn_quick.py

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python tune_hybrid_gcn_quick.py
```

### 5. Resume from Checkpoint
If experiments are interrupted, check `results.json` to see which configs were already tested, then modify the script to skip them.

## Expected Improvements

Based on typical hyperparameter tuning results:

- **Quick wins** (+1-3%): Adjusting learning rate and weight decay
- **Moderate gains** (+3-7%): Finding better hidden_dim and num_layers
- **Significant gains** (+7%+): Optimal combination of multiple parameters

Goal: Find a configuration with val@20 > 0.24 (currently 0.2349)

## Common Issues

### OOM (Out of Memory)
**Solution:** Reduce batch_size or hidden_dim in SEARCH_SPACE

### Training Too Slow
**Solution:**
- Use quick version first
- Reduce number of experiments
- Use focused search instead of random search

### All Results Similar to Baseline
**Solution:**
- Expand search space (try more extreme values)
- Check if data loading is consistent
- Verify model is actually using different configs

### Early Stopping Too Soon
**Solution:** Increase patience in FIXED_PARAMS

## Next Steps After Tuning

1. **Train best config multiple times** (different random seeds) to verify stability
2. **Test on other datasets** to check generalization
3. **Try ensemble methods** combining top configs
4. **Explore advanced techniques**:
   - Learning rate scheduling
   - Gradient clipping
   - Different optimizers (Adam vs AdamW vs SGD)
   - Architecture variations (GAT, GraphSAGE, etc.)

## Questions?

If results are unclear or you need help interpreting the analysis, review:
- The parameter impact analysis section (shows avg performance per param value)
- The top configurations (look for common patterns)
- The improvement distribution (are improvements consistent or random?)

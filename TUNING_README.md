# Hyperparameter Tuning Guide

This guide explains how to use the hyperparameter search tools for baseline models.

## Quick Start

```bash
# Run minimal hyperparameter search on GCN
python tune_baselines.py

# Analyze results
python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json

# Test a specific hypothesis
python run_focused_experiment.py --model GCN --vary dropout --values 0.0 0.1 0.2
```

## Files Overview

### 1. `tune_baselines.py`
**Purpose**: Comprehensive grid search over hyperparameter space

**Features**:
- Three search modes: minimal, standard, extensive
- Supports all 4 baseline models (GCN, GraphSAGE, GAT, GraphTransformer)
- Logs all results to JSON for later analysis
- Automatic best configuration detection

**Configuration**:
Edit these variables in the script:
```python
SEARCH_TYPE = 'minimal'  # 'minimal', 'standard', or 'extensive'
MODELS_TO_SEARCH = ['GCN']  # Add other models: 'GraphSAGE', 'GAT', 'GraphTransformer'
```

**Usage**:
```bash
# Run with default settings (minimal search on GCN)
python tune_baselines.py

# Monitor progress
tail -f logs/hyperparam_search_TIMESTAMP/search.log
```

**Output**:
- `logs/hyperparam_search_TIMESTAMP/search.log` - Detailed training logs
- `logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json` - All results

**Time estimates**:
- Minimal: 2-4 hours per model
- Standard: 1-2 days per model
- Extensive: 3-5 days per model

### 2. `analyze_hyperparam_results.py`
**Purpose**: Comprehensive analysis of search results

**Features**:
- Statistical summary of all configurations
- Parameter impact analysis
- Best configuration per model
- Generalization analysis (val-test gap)
- Top performing configurations

**Usage**:
```bash
# Basic analysis
python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json

# Export best configs to file
python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json \
    --export best_configs.json
```

**Output**:
Prints to console:
- Overall performance statistics
- Model-specific analysis
- Top 10 configurations
- Top 5 generalizing configs
- Parameter impact analysis

### 3. `run_focused_experiment.py`
**Purpose**: Test specific hypotheses without full grid search

**Features**:
- Vary a single parameter
- Test specific configurations
- Use custom base configurations
- Quick iteration on ideas

**Usage Examples**:

```bash
# Does dropout help GCN?
python run_focused_experiment.py --model GCN --vary dropout --values 0.0 0.1 0.2 0.3

# What's the best hidden dimension for GraphSAGE?
python run_focused_experiment.py --model GraphSAGE --vary hidden_dim --values 64 128 256 512

# Does multi-strategy decoder help?
python run_focused_experiment.py --model GCN --vary use_multi_strategy --values False True

# Test a specific configuration
python run_focused_experiment.py --model GCN --config '{"hidden_dim": 256, "dropout": 0.1, "lr": 0.005}'

# Test with custom base config
python run_focused_experiment.py --model GCN --vary lr --values 0.001 0.005 0.01 \
    --base-config my_base_config.json
```

**Output**:
- `logs/focused_experiment_TIMESTAMP/experiment.log` - Training logs
- `logs/focused_experiment_TIMESTAMP/results.json` - Results

### 4. `docs/hyperparameter_experiments.md`
**Purpose**: Detailed experimental design documentation

**Contents**:
- Motivation and goals
- Search space definitions
- Parameter descriptions and hypotheses
- Experimental protocol
- Analysis plan
- Expected outcomes

**Use this for**:
- Understanding the search strategy
- Planning experiments
- Interpreting results

## Typical Workflow

### Phase 1: Quick Exploration (Minimal Search)

Test all models with minimal search to identify the most promising:

```bash
# Edit tune_baselines.py
SEARCH_TYPE = 'minimal'
MODELS_TO_SEARCH = ['GCN', 'GraphSAGE', 'GAT', 'GraphTransformer']

# Run
python tune_baselines.py

# Analyze
python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json
```

**Goal**: Identify which model(s) to focus on for deeper search.

### Phase 2: Hypothesis Testing

Based on minimal search results, test specific hypotheses:

```bash
# Example: GraphSAGE was best, does batch norm help?
python run_focused_experiment.py --model GraphSAGE --vary use_batch_norm --values False True

# Example: Test dropout impact on best configuration
python run_focused_experiment.py --model GraphSAGE --vary dropout --values 0.0 0.05 0.1 0.15 0.2

# Example: Learning rate sensitivity
python run_focused_experiment.py --model GraphSAGE --vary lr --values 0.001 0.005 0.01 0.02
```

**Goal**: Understand parameter sensitivity and find optimal values.

### Phase 3: Standard Search on Best Model

Run comprehensive search on the best model from Phase 1:

```bash
# Edit tune_baselines.py
SEARCH_TYPE = 'standard'
MODELS_TO_SEARCH = ['GraphSAGE']  # Your best model

# Run
python tune_baselines.py

# Analyze
python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json \
    --export optimal_configs.json
```

**Goal**: Find the absolute best configuration.

### Phase 4: Validation

Re-run best configs multiple times to measure variance:

```bash
# Extract best config and run 5 times
for i in {1..5}; do
    python run_focused_experiment.py --model GraphSAGE \
        --config '{"hidden_dim": 256, "num_layers": 3, "dropout": 0.1, "lr": 0.005}'
done
```

**Goal**: Ensure results are stable and reproducible.

## Common Scenarios

### Scenario 1: "I want to quickly see if any model can beat GraphSAGE's 17%"

```bash
# Run minimal search on all models
python tune_baselines.py  # With MODELS_TO_SEARCH = ['GCN', 'GraphSAGE', 'GAT', 'GraphTransformer']

# Check summary
tail -n 50 logs/hyperparam_search_TIMESTAMP/search.log
```

### Scenario 2: "I want to optimize GCN specifically"

```bash
# Start with standard search
python tune_baselines.py  # With SEARCH_TYPE='standard', MODELS_TO_SEARCH=['GCN']

# Analyze results
python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json

# Test refinements based on analysis
python run_focused_experiment.py --model GCN --vary hidden_dim --values 128 192 256
```

### Scenario 3: "I have a hypothesis about attention heads"

```bash
# Test quickly
python run_focused_experiment.py --model GAT --vary heads --values 2 4 8 16

# If promising, do grid search with other params
# Edit tune_baselines.py to focus on GAT with wider head range
```

### Scenario 4: "I want to reduce the val-test gap"

```bash
# Run standard search
python tune_baselines.py

# Analyze for best generalizing configs
python analyze_hyperparam_results.py logs/hyperparam_search_TIMESTAMP/hyperparam_search_results.json

# Check "TOP 5 BEST GENERALIZING CONFIGS" section
# Test those configs with higher weight decay
python run_focused_experiment.py --model GCN --vary weight_decay \
    --values 1e-4 5e-4 1e-3 5e-3 --base-config best_generalizing_config.json
```

## Parallel Execution

If you have multiple GPUs or machines:

### Option 1: Multiple Models in Parallel

```bash
# Terminal 1
python tune_baselines.py  # MODELS_TO_SEARCH = ['GCN']

# Terminal 2
python tune_baselines.py  # MODELS_TO_SEARCH = ['GraphSAGE']

# Terminal 3
python tune_baselines.py  # MODELS_TO_SEARCH = ['GAT']

# Terminal 4
python tune_baselines.py  # MODELS_TO_SEARCH = ['GraphTransformer']
```

### Option 2: Split Search Space

Create multiple versions of `tune_baselines.py` with different parameter ranges and run in parallel.

## Tips and Tricks

### 1. Start Small
Always run a few configs first to ensure everything works:
```python
# In tune_baselines.py, temporarily limit search space
'hidden_dim': [128],  # Instead of [64, 128, 256, 512]
'num_layers': [2],    # Instead of [2, 3, 4]
```

### 2. Monitor GPU Memory
If you get CUDA OOM errors:
- Reduce `batch_size`
- Reduce `hidden_dim`
- Reduce `num_layers`

### 3. Check Early Results
Don't wait for full search. Check logs after first few configs:
```bash
tail -f logs/hyperparam_search_TIMESTAMP/search.log | grep "Test Hits@20"
```

### 4. Resume Failed Searches
If a search crashes:
1. Note which configs completed (check JSON file)
2. Modify search space to exclude completed configs
3. Restart and combine results later

### 5. Use Focused Experiments for Debugging
If results seem off:
```bash
# Test baseline config
python run_focused_experiment.py --model GCN --config '{}'  # Uses defaults

# Compare with specific change
python run_focused_experiment.py --model GCN --config '{"dropout": 0.1}'
```

## Interpreting Results

### Good Signs
- Test Hits@20 > 15% (for baselines)
- Val-test gap < 3% (good generalization)
- Best epoch < 100 (efficient training)
- Consistent performance across similar configs

### Red Flags
- Val-test gap > 5% (overfitting)
- Best epoch = max epochs (needs more training or stuck)
- Huge variance across similar configs (unstable)
- Test < 10% (something wrong)

### Common Patterns

**Pattern**: High dropout hurts performance
- **Cause**: Dense graphs don't need aggressive regularization
- **Action**: Keep dropout ≤ 0.2

**Pattern**: Larger hidden_dim helps
- **Cause**: Complex drug interaction patterns need capacity
- **Action**: Try 256 or 512 if memory allows

**Pattern**: More layers help initially, then hurt
- **Cause**: Over-smoothing on dense graphs
- **Action**: Keep layers ≤ 4

**Pattern**: Multi-strategy decoder inconsistent
- **Cause**: More parameters need more data/regularization
- **Action**: Test with higher weight_decay

## Troubleshooting

### Problem: Search is too slow
**Solutions**:
1. Use 'minimal' search mode
2. Reduce `patience` (faster early stopping)
3. Increase `eval_every` (less frequent evaluation)
4. Run models in parallel
5. Use focused experiments instead

### Problem: All configs perform poorly
**Check**:
1. Is data loading correctly? (check num_nodes, edge counts)
2. Is dropout=0 enforced? (minimal trainer checks this)
3. Are edge splits correct? (train/val/test)
4. Is evaluation function working? (test with known good config)

### Problem: Can't analyze results
**Check**:
1. Is JSON file valid? `python -m json.tool <file>`
2. Are there any results? `cat <file> | grep best_test_hits`
3. Did experiments complete? Check log file

### Problem: Inconsistent results
**Possible causes**:
1. Random seed not set (results will vary)
2. Batch size too small (noisy gradients)
3. Evaluation batch size different (numerical differences)

**Fix**: Re-run best configs 3-5 times to measure variance

## Advanced Usage

### Custom Search Spaces

Edit `create_search_space()` in `tune_baselines.py`:

```python
def create_search_space(search_type='custom'):
    if search_type == 'custom':
        return {
            'hidden_dim': [192, 224, 256],  # Fine-grained around 256
            'num_layers': [2, 3],
            'dropout': [0.08, 0.1, 0.12],   # Fine-grained around 0.1
            'decoder_dropout': [0.0],
            'use_multi_strategy': [True],    # Only test multi-strategy
            'lr': [0.005],                   # Fixed based on prior results
            'weight_decay': [1e-4, 5e-4, 1e-3],
            'batch_size': [50000],
        }
```

### Custom Base Configs

Create `my_base_config.json`:
```json
{
  "hidden_dim": 256,
  "num_layers": 3,
  "dropout": 0.1,
  "decoder_dropout": 0.0,
  "use_multi_strategy": false,
  "lr": 0.005,
  "weight_decay": 1e-4,
  "batch_size": 50000,
  "epochs": 200,
  "patience": 20
}
```

Use with focused experiments:
```bash
python run_focused_experiment.py --model GCN --vary dropout --values 0.05 0.1 0.15 \
    --base-config my_base_config.json
```

### Combining Results

If you ran multiple searches:
```python
import json

results = []
for file in ['results1.json', 'results2.json', 'results3.json']:
    with open(file) as f:
        results.extend(json.load(f))

with open('combined_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Analyze
python analyze_hyperparam_results.py combined_results.json
```

## Next Steps After Tuning

Once you find optimal configurations:

1. **Update `train_baselines.py`** with the best configs
2. **Document findings** in project documentation
3. **Run multiple seeds** (3-5 runs) to measure variance
4. **Compare with SEAL** and other advanced methods
5. **Consider ensemble methods** combining multiple configs
6. **Publish results** with confidence intervals

## Questions?

Check the detailed documentation:
- `docs/hyperparameter_experiments.md` - Experiment design
- `README.md` - Project overview
- Source code comments - Implementation details

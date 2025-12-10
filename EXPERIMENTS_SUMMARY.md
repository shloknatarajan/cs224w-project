# Hyperparameter Search Experiments - Summary

## What's Been Created

I've designed a comprehensive hyperparameter search system for your minimal baseline models. Here's what you have:

### 🔧 Core Tools

1. **`tune_baselines.py`** - Main grid search engine
   - Three search modes (minimal/standard/extensive)
   - Supports all 4 models (GCN, GraphSAGE, GAT, GraphTransformer)
   - Automatic logging and result collection

2. **`analyze_hyperparam_results.py`** - Statistical analysis tool
   - Parameter impact analysis
   - Model comparisons
   - Generalization metrics
   - Top configuration detection

3. **`run_focused_experiment.py`** - Hypothesis testing tool
   - Quick single-parameter sweeps
   - Specific configuration testing
   - Custom base configurations

4. **`visualize_results.py`** - Visualization tool
   - Box plots for parameter impact
   - Performance distributions
   - Pareto frontiers
   - Parameter interaction heatmaps

### 📚 Documentation

1. **`TUNING_README.md`** - Complete usage guide
   - Quick start examples
   - Typical workflows
   - Troubleshooting tips
   - Advanced usage patterns

2. **`docs/hyperparameter_experiments.md`** - Experimental design
   - Search strategy rationale
   - Parameter descriptions and hypotheses
   - Expected outcomes
   - Analysis plan

3. **`EXPERIMENTS_SUMMARY.md`** - This file!

## Quick Start

```bash
# 1. Run minimal search on GCN (2-4 hours)
python tune_baselines.py

# 2. Analyze results
python analyze_hyperparam_results.py logs/hyperparam_search_*/hyperparam_search_results.json

# 3. Visualize (requires matplotlib/seaborn)
python visualize_results.py logs/hyperparam_search_*/hyperparam_search_results.json

# 4. Test specific hypothesis
python run_focused_experiment.py --model GCN --vary dropout --values 0.0 0.1 0.2
```

## Search Strategies

### Minimal Search (Recommended First)
- **Time**: 2-4 hours per model
- **Configs**: 36-72 per model
- **Goal**: Quick validation of promising parameter ranges
- **Use when**: Exploring multiple models or limited time

### Standard Search
- **Time**: 1-2 days per model
- **Configs**: 288+ per model
- **Goal**: Thorough exploration with decoder variants
- **Use when**: Optimizing specific model(s)

### Extensive Search
- **Time**: 3-5 days per model
- **Configs**: 1000+ per model
- **Goal**: Find absolute best configuration
- **Use when**: Final optimization of chosen model

## Key Parameters Being Searched

### Architecture
- **hidden_dim**: [64, 128, 256, 512] - Model capacity
- **num_layers**: [2, 3, 4, 5] - Depth (watch for over-smoothing)
- **dropout**: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3] - Regularization
- **decoder_dropout**: [0.0, 0.05, 0.1, 0.2] - Link prediction head regularization

### Decoder Strategy
- **use_multi_strategy**: [False, True] - Simple vs. complex edge embeddings

### Model-Specific
- **heads**: [2, 4, 8, 16] - Attention heads (GAT/Transformer)
- **use_batch_norm**: [False, True] - Batch normalization (GraphSAGE)

### Training
- **lr**: [0.0005, 0.001, 0.005, 0.01, 0.02] - Learning rate
- **weight_decay**: [0.0, 1e-5, 1e-4, 1e-3] - L2 regularization
- **batch_size**: [25000, 50000, 100000] - Batch size

## Expected Performance Improvements

Current baselines → Target improvements:

| Model | Current | Target | Strategy |
|-------|---------|--------|----------|
| **GCN** | 13-24% | >25% | Test higher capacity, small dropout |
| **GraphSAGE** | 17% | >20% | Best baseline, optimize batch norm & layers |
| **GAT** | 11% | >15% | Optimize attention heads, try larger hidden_dim |
| **GraphTransformer** | 8% | >12% | Similar to GAT, may need lower LR |

## Recommended Workflow

### Week 1: Initial Exploration
```bash
# Day 1-2: Run minimal search on all models
python tune_baselines.py  # Edit to test each model

# Day 3: Analyze and identify best model
python analyze_hyperparam_results.py <results_file>

# Day 4-5: Test hypotheses on best model
python run_focused_experiment.py --model <best> --vary <param> --values ...
```

### Week 2: Deep Dive
```bash
# Run standard search on top 2 models
python tune_baselines.py  # Edit SEARCH_TYPE='standard'

# Analyze and visualize
python analyze_hyperparam_results.py <results_file>
python visualize_results.py <results_file>
```

### Week 3: Final Optimization
```bash
# Extensive search on best model (optional)
python tune_baselines.py  # Edit SEARCH_TYPE='extensive'

# Validate best configs (run 3-5 times)
python run_focused_experiment.py --model <best> --config '<best_config>'

# Update train_baselines.py with optimal settings
```

## What to Look For in Results

### Success Indicators ✅
- Test Hits@20 > 15% (baseline level)
- Val-Test gap < 3% (good generalization)
- Best epoch < 100 (efficient training)
- Consistent performance across similar configs

### Warning Signs ⚠️
- Val-Test gap > 5% (overfitting)
- Best epoch at max (needs more training)
- High variance across similar configs
- Test < 10% (something wrong)

### Common Patterns
1. **Dropout**: Small amounts (0.1-0.2) may help; high dropout (>0.3) hurts
2. **Hidden dim**: Larger often better (256-512) for complex DDI patterns
3. **Layers**: 2-3 optimal; more causes over-smoothing
4. **Learning rate**: Lower for attention models (0.005 vs 0.01)
5. **Multi-strategy**: Inconsistent; needs tuning with other params

## Analysis Outputs

### From `analyze_hyperparam_results.py`:
- Overall statistics (mean, std, best, worst)
- Model-specific analysis
- Top 10 configurations by test performance
- Top 5 generalizing configurations (low val-test gap)
- Parameter impact analysis (which params matter most)

### From `visualize_results.py`:
- `performance_distribution.png` - Histogram and box plot
- `model_comparison.png` - Compare across architectures
- `generalization_analysis.png` - Val-test gap patterns
- `pareto_frontier.png` - Trade-off between performance and generalization
- `training_efficiency.png` - Epochs vs performance
- `impact_<param>.png` - Box plots for each parameter
- `heatmap_<param1>_vs_<param2>.png` - Parameter interactions

## Tips for Success

1. **Start Small**: Always begin with minimal search to validate setup
2. **Monitor Early**: Check first few configs to catch issues
3. **Use Focused Experiments**: Test hypotheses quickly before big searches
4. **Run in Parallel**: Use multiple GPUs/terminals for different models
5. **Document Findings**: Keep notes on interesting patterns
6. **Check Stability**: Re-run best configs 3-5 times

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 25000 or lower
- Reduce `hidden_dim` to 64 or 128
- Reduce `num_layers` to 2

### Slow Training
- Increase `eval_every` to 10
- Reduce `patience` to 15
- Use 'minimal' search mode

### Poor Results
- Verify data loading (check num_nodes, edges)
- Ensure dropout=0 in minimal configs
- Test with known good config first

### Can't Run Visualizations
```bash
# Install dependencies
pip install matplotlib seaborn

# Or use analysis script only (no plots needed)
python analyze_hyperparam_results.py <results_file>
```

## Files Reference

```
cs224w-project/
├── tune_baselines.py                    # Main grid search
├── analyze_hyperparam_results.py        # Statistical analysis
├── run_focused_experiment.py            # Quick hypothesis testing
├── visualize_results.py                 # Create plots
├── TUNING_README.md                     # Detailed usage guide
├── EXPERIMENTS_SUMMARY.md               # This file
├── docs/
│   └── hyperparameter_experiments.md    # Experiment design
└── logs/
    └── hyperparam_search_TIMESTAMP/
        ├── search.log                   # Training logs
        ├── hyperparam_search_results.json  # All results
        └── plots/                       # Visualizations
            ├── performance_distribution.png
            ├── model_comparison.png
            ├── pareto_frontier.png
            └── ...
```

## Next Steps

1. **Read** `TUNING_README.md` for detailed usage instructions
2. **Run** minimal search on one model to test the system
3. **Analyze** results and identify promising configurations
4. **Expand** to more models or deeper search based on findings
5. **Validate** best configs with multiple runs
6. **Document** findings and update baseline scripts

## Questions?

- **Usage**: See `TUNING_README.md`
- **Design rationale**: See `docs/hyperparameter_experiments.md`
- **Source code**: Check comments in each script
- **Baseline models**: See `src/models/baselines/`
- **Training logic**: See `src/training/minimal_trainer.py`

Good luck with your experiments! 🚀

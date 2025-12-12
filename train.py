#!/usr/bin/env python
"""
Unified training script for all models in the 224w-project.

Supports:
- Baseline GNNs: GCN, GraphSAGE, GraphTransformer, GAT
- Advanced models: GDINN, GCNAdvanced, Hybrid models
- External features: Morgan, PubChem, ChemBERTa, Drug-Targets

Usage:
    # Train baseline GCN
    python train.py --model gcn

    # Train GDINN with all external features
    python train.py --model gdinn --external-features all

    # Train GCNAdvanced (structure-only)
    python train.py --model gcn-advanced

    # Train GraphSAGE with specific features
    python train.py --model sage --external-features morgan,chemberta

    # Train with custom hyperparameters
    python train.py --model gcn --epochs 300 --lr 0.001 --hidden-dim 256
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_dataset
from src.data.external_features import FeatureConfig, load_external_features
from src.models.baselines import GCN, GraphSAGE, GraphTransformer, GAT
from src.models.advanced import GDINN, GCNAdvanced
from src.models.advanced.gdinn import LinkPredictor, train_with_external, test_with_external
from src.models.advanced.gcn_advanced import (
    LinkPredictor as GCNAdvancedPredictor,
    train as train_gcn_advanced_epoch,
    test as test_gcn_advanced_epoch,
)
from src.models.chemberta_baselines import ChemBERTaGCN, ChemBERTaGraphSAGE, ChemBERTaGAT, ChemBERTaGraphTransformer
from src.models.morgan_baselines import MorganGCN
from src.models.hybrid import HybridGCN, HybridGraphSAGE, HybridGraphTransformer, HybridGAT
from src.training import train_minimal_baseline
from src.evals import evaluate


# Model registry
BASELINE_MODELS = ['gcn', 'sage', 'transformer', 'gat']
CHEMISTRY_MODELS = ['chemberta-gcn', 'chemberta-sage', 'chemberta-gat', 'chemberta-transformer', 'morgan-gcn']
HYBRID_MODELS = ['hybrid-gcn', 'hybrid-sage', 'hybrid-transformer', 'hybrid-gat']
ADVANCED_MODELS = ['gdinn', 'gcn-advanced']
ALL_MODELS = BASELINE_MODELS + CHEMISTRY_MODELS + HYBRID_MODELS + ADVANCED_MODELS


def setup_logging(args):
    """Set up logging to file and console."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build log directory name
    features_suffix = ""
    if args.external_features:
        if args.external_features == 'all':
            features_suffix = "_all_features"
        else:
            features = args.external_features.replace(',', '_')
            features_suffix = f"_{features}"

    log_dir = f'logs/{args.model}{features_suffix}_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{args.model}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_dir, log_file


def get_device(args):
    """Get the appropriate device."""
    if args.device is not None:
        if args.device == 'cpu':
            return torch.device('cpu')
        elif args.device == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                logging.warning("CUDA not available, falling back to CPU")
                return torch.device('cpu')
        elif args.device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                logging.warning("MPS not available, falling back to CPU")
                return torch.device('cpu')

    # Auto-detect
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def parse_external_features(feature_str):
    """Parse external features string into config flags."""
    if not feature_str or feature_str.lower() == 'none':
        return {
            'use_morgan': False,
            'use_pubchem': False,
            'use_chemberta': False,
            'use_drug_targets': False
        }

    if feature_str.lower() == 'all':
        return {
            'use_morgan': True,
            'use_pubchem': True,
            'use_chemberta': True,
            'use_drug_targets': True
        }

    features = [f.strip().lower() for f in feature_str.split(',')]
    return {
        'use_morgan': 'morgan' in features,
        'use_pubchem': 'pubchem' in features,
        'use_chemberta': 'chemberta' in features,
        'use_drug_targets': 'drug-targets' in features or 'drug_targets' in features
    }


def load_external_features_if_needed(args, num_nodes, logger):
    """Load external features if specified."""
    feature_config = parse_external_features(args.external_features)

    if not any(feature_config.values()):
        logger.info("Training without external features")
        return None, {}

    logger.info(f"Loading external features: {[k for k, v in feature_config.items() if v]}")

    data_dir = Path(args.data_dir)
    config = FeatureConfig(
        use_morgan=feature_config['use_morgan'],
        use_pubchem=feature_config['use_pubchem'],
        use_chemberta=feature_config['use_chemberta'],
        use_drug_targets=feature_config['use_drug_targets'],
        morgan_path=data_dir / "morgan_features.pt",
        pubchem_path=data_dir / "ogbl_ddi_properties.csv",
        chemberta_path=data_dir / "chemberta_embeddings.pt",
        drug_targets_path=data_dir / "drug_target_features.pt",
    )

    ext_features = load_external_features(config, num_nodes)

    if ext_features.total_dim == 0:
        logger.error("No external features could be loaded!")
        logger.error("Run the data fetching scripts first:")
        logger.error("  python -m src.data.fetch_smiles")
        logger.error("  python -m src.data.fetch_pubchem_properties")
        logger.error("  python -m src.data.extract_chemberta_embeddings")
        logger.error("  python -m src.data.fetch_drug_targets")
        sys.exit(1)

    logger.info(f"Loaded features with dimensions: {ext_features.feature_dims}")
    logger.info(f"Total feature dimension: {ext_features.total_dim}")

    # Convert to dict for trainers
    external_features_dict = {}
    if ext_features.morgan is not None:
        external_features_dict['morgan'] = ext_features.morgan
    if ext_features.pubchem is not None:
        external_features_dict['pubchem'] = ext_features.pubchem
    if ext_features.chemberta is not None:
        external_features_dict['chemberta'] = ext_features.chemberta
    if ext_features.drug_targets is not None:
        external_features_dict['drug_targets'] = ext_features.drug_targets

    return external_features_dict, ext_features.feature_dims


def create_model(args, num_nodes, external_dims, chemberta_features, morgan_features, device, logger):
    """Create model based on args."""
    model_name = args.model.lower()

    if model_name in BASELINE_MODELS:
        # Baseline models
        if model_name == 'gcn':
            model = GCN(
                num_nodes,
                args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )
        elif model_name == 'sage':
            model = GraphSAGE(
                num_nodes,
                args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )
        elif model_name == 'transformer':
            model = GraphTransformer(
                num_nodes,
                args.hidden_dim,
                num_layers=args.num_layers,
                heads=args.attention_heads,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )
        elif model_name == 'gat':
            model = GAT(
                num_nodes,
                args.hidden_dim,
                num_layers=args.num_layers,
                heads=args.attention_heads,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )

        logger.info(f"Created {model_name.upper()} model:")
        logger.info(f"  hidden_dim: {args.hidden_dim}")
        logger.info(f"  num_layers: {args.num_layers}")
        logger.info(f"  dropout: {args.dropout}")

        return model

    elif model_name in CHEMISTRY_MODELS:
        # Models with chemistry features as input
        if model_name == 'chemberta-gcn':
            if chemberta_features is None:
                logger.error("ChemBERTa features not loaded. Use --external-features chemberta")
                sys.exit(1)
            model = ChemBERTaGCN(
                in_channels=chemberta_features.shape[1],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )
        elif model_name == 'chemberta-sage':
            if chemberta_features is None:
                logger.error("ChemBERTa features not loaded. Use --external-features chemberta")
                sys.exit(1)
            model = ChemBERTaGraphSAGE(
                in_channels=chemberta_features.shape[1],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )
        elif model_name == 'chemberta-gat':
            if chemberta_features is None:
                logger.error("ChemBERTa features not loaded. Use --external-features chemberta")
                sys.exit(1)
            model = ChemBERTaGAT(
                in_channels=chemberta_features.shape[1],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                heads=args.attention_heads,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )
        elif model_name == 'chemberta-transformer':
            if chemberta_features is None:
                logger.error("ChemBERTa features not loaded. Use --external-features chemberta")
                sys.exit(1)
            model = ChemBERTaGraphTransformer(
                in_channels=chemberta_features.shape[1],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                heads=args.attention_heads,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )
        elif model_name == 'morgan-gcn':
            if morgan_features is None:
                logger.error("Morgan features not loaded. Use --external-features morgan")
                sys.exit(1)
            model = MorganGCN(
                in_channels=morgan_features.shape[1],
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                decoder_dropout=args.decoder_dropout
            )

        logger.info(f"Created {model_name.upper()} model:")
        logger.info(f"  hidden_dim: {args.hidden_dim}")
        logger.info(f"  num_layers: {args.num_layers}")
        logger.info(f"  dropout: {args.dropout}")

        return model

    elif model_name == 'gdinn':
        # GDINN with external features
        if not external_dims:
            logger.error("GDINN requires external features. Use --external-features")
            sys.exit(1)

        model = GDINN(
            num_nodes=num_nodes,
            hidden_channels=args.hidden_dim,
            out_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            external_dims=external_dims,
            fusion=args.fusion,
        )

        logger.info(f"Created GDINN model:")
        logger.info(f"  hidden_channels: {args.hidden_dim}")
        logger.info(f"  num_layers: {args.num_layers}")
        logger.info(f"  fusion: {args.fusion}")
        logger.info(f"  external_dims: {external_dims}")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    elif model_name == 'gcn-advanced':
        # GCNAdvanced (structure-only model)
        model = GCNAdvanced(
            args.hidden_dim,
            args.hidden_dim,
            args.hidden_dim,
            args.num_layers,
            args.dropout,
        )

        logger.info(f"Created GCNAdvanced model:")
        logger.info(f"  hidden_channels: {args.hidden_dim}")
        logger.info(f"  num_layers: {args.num_layers}")
        logger.info(f"  dropout: {args.dropout}")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    elif model_name in HYBRID_MODELS:
        # Hybrid models require external features for the chemistry-aware decoder
        if not external_dims:
            logger.warning("Hybrid models work best with external features for chemistry-aware decoder")

        # Select hybrid variant
        if model_name == 'hybrid-gcn':
            model = HybridGCN(
                num_nodes,
                args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                chemical_dim=sum(external_dims.values()) if external_dims else 0
            )
        elif model_name == 'hybrid-sage':
            model = HybridGraphSAGE(
                num_nodes,
                args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                chemical_dim=sum(external_dims.values()) if external_dims else 0
            )
        elif model_name == 'hybrid-transformer':
            model = HybridGraphTransformer(
                num_nodes,
                args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                num_heads=args.attention_heads,
                chemical_dim=sum(external_dims.values()) if external_dims else 0
            )
        elif model_name == 'hybrid-gat':
            model = HybridGAT(
                num_nodes,
                args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                num_heads=args.attention_heads,
                chemical_dim=sum(external_dims.values()) if external_dims else 0
            )

        logger.info(f"Created {model_name.upper()} model:")
        logger.info(f"  hidden_dim: {args.hidden_dim}")
        logger.info(f"  num_layers: {args.num_layers}")
        logger.info(f"  chemical_dim: {sum(external_dims.values()) if external_dims else 0}")

        return model

    else:
        logger.error(f"Unknown model: {model_name}")
        sys.exit(1)


def train_baseline_model(args, model, data, splits, chemberta_features, morgan_features, device, logger):
    """Train baseline models using minimal trainer."""
    train_pos, valid_pos, valid_neg, test_pos, test_neg = splits

    # Determine which node features to use
    node_features = None
    if args.model in CHEMISTRY_MODELS:
        if 'chemberta' in args.model:
            node_features = chemberta_features
        elif 'morgan' in args.model:
            node_features = morgan_features

    # Modify data object if using node features
    if node_features is not None:
        data.x = node_features
        logger.info(f"Using node features: {node_features.shape}")

    # Need to get evaluator
    from ogb.linkproppred import Evaluator
    evaluator = Evaluator(name='ogbl-ddi')

    # Hybrid models have internal dropout in decoders, so skip the check for them
    skip_dropout = args.model in HYBRID_MODELS

    result = train_minimal_baseline(
        name=args.model.upper(),
        model=model,
        data=data,
        train_pos=train_pos,
        valid_pos=valid_pos,
        valid_neg=valid_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        evaluate_fn=lambda m, p, n, b: evaluate(m, data, evaluator, p, n, b),
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        skip_dropout_check=skip_dropout,
    )

    return result


def train_gdinn_model(args, model, data, splits, external_features, device, logger):
    """Train GDINN model using native training functions."""
    import torch_geometric.transforms as T
    from torch_geometric.data import Data
    from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
    from dataclasses import dataclass

    @dataclass
    class GDINNResult:
        best_val_hits: float
        best_test_hits: float
        best_epoch: int

    # Load dataset with SparseTensor transform (required by GDINN)
    logger.info("Loading dataset with SparseTensor format for GDINN...")
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    sparse_data = dataset[0]
    adj_t = sparse_data.adj_t.to(device)
    split_edge = dataset.get_edge_split()
    num_nodes = sparse_data.num_nodes

    # Create eval_train subset
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[: split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    # Move external features to device
    ext_features_device = None
    if external_features:
        ext_features_device = {
            k: v.to(device) if v is not None else None
            for k, v in external_features.items()
        }

    # Move model to device and create predictor
    model = model.to(device)
    predictor = LinkPredictor(
        args.hidden_dim, args.hidden_dim, 1, args.num_layers, args.dropout
    ).to(device)

    evaluator = Evaluator(name='ogbl-ddi')

    # Optimizer for both model and predictor
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )

    logger.info("=" * 80)
    logger.info("Starting GDINN Training")
    logger.info(f"  epochs: {args.epochs}")
    logger.info(f"  lr: {args.lr}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  eval_every: {args.eval_every}")
    logger.info(f"  patience: {args.patience}")
    logger.info("=" * 80)

    best_valid = 0.0
    best_test = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_with_external(
            model,
            predictor,
            adj_t,
            split_edge,
            optimizer,
            args.batch_size,
            ext_features_device,
        )

        if epoch % args.eval_every == 0:
            scores = test_with_external(
                model, predictor, adj_t, split_edge, evaluator,
                args.batch_size, ext_features_device
            )

            # Extract Hits@20 (main metric)
            _, valid_hits, test_hits = scores["Hits@20"]

            if valid_hits > best_valid:
                best_valid = valid_hits
                best_test = test_hits
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            logger.info(
                f"[GDINN] Epoch {epoch:04d} | "
                f"loss {loss:.4f} | "
                f"val@20 {valid_hits:.4f} | "
                f"test@20 {test_hits:.4f} | "
                f"best {best_valid:.4f} (ep {best_epoch})"
            )

            # Early stopping
            if args.patience and epochs_no_improve >= args.patience:
                logger.info(
                    f"[GDINN] Early stopping at epoch {epoch} "
                    f"(no val improvement for {epochs_no_improve} evals)"
                )
                break

    logger.info(
        f"[GDINN] Done. Best val@20={best_valid:.4f} | test@20={best_test:.4f} "
        f"(epoch {best_epoch})"
    )

    return GDINNResult(best_valid, best_test, best_epoch)


def train_gcn_advanced_model(args, model, data, splits, device, logger):
    """Train GCNAdvanced model using OGB reference training functions."""
    import torch_geometric.transforms as T
    from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
    from dataclasses import dataclass

    @dataclass
    class GCNAdvancedResult:
        best_val_hits: float
        best_test_hits: float
        best_epoch: int

    # Load dataset with SparseTensor transform (required by GCNAdvanced)
    logger.info("Loading dataset with SparseTensor format for GCNAdvanced...")
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    sparse_data = dataset[0]
    adj_t = sparse_data.adj_t.to(device)
    split_edge = dataset.get_edge_split()
    num_nodes = sparse_data.num_nodes

    # Create eval_train subset (matching OGB reference)
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[: split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    # Move model to device and create embedding + predictor
    model = model.to(device)
    emb = torch.nn.Embedding(num_nodes, args.hidden_dim).to(device)
    predictor = GCNAdvancedPredictor(
        args.hidden_dim, args.hidden_dim, 1, args.num_layers, args.dropout
    ).to(device)

    # Initialize parameters
    torch.nn.init.xavier_uniform_(emb.weight)
    model.reset_parameters()
    predictor.reset_parameters()

    evaluator = Evaluator(name='ogbl-ddi')

    # Optimizer for model, embedding, and predictor
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(emb.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )

    logger.info("=" * 80)
    logger.info("Starting GCNAdvanced Training")
    logger.info(f"  epochs: {args.epochs}")
    logger.info(f"  lr: {args.lr}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  eval_every: {args.eval_every}")
    logger.info(f"  patience: {args.patience}")
    logger.info("=" * 80)

    best_valid = 0.0
    best_test = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_gcn_advanced_epoch(
            model,
            predictor,
            emb.weight,
            adj_t,
            split_edge,
            optimizer,
            args.batch_size,
        )

        if epoch % args.eval_every == 0:
            scores = test_gcn_advanced_epoch(
                model, predictor, emb.weight, adj_t, split_edge, evaluator,
                args.batch_size
            )

            # Extract Hits@20 (main metric)
            _, valid_hits, test_hits = scores["Hits@20"]

            if valid_hits > best_valid:
                best_valid = valid_hits
                best_test = test_hits
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            logger.info(
                f"[GCNAdvanced] Epoch {epoch:04d} | "
                f"loss {loss:.4f} | "
                f"val@20 {valid_hits:.4f} | "
                f"test@20 {test_hits:.4f} | "
                f"best {best_valid:.4f} (ep {best_epoch})"
            )

            # Early stopping
            if args.patience and epochs_no_improve >= args.patience:
                logger.info(
                    f"[GCNAdvanced] Early stopping at epoch {epoch} "
                    f"(no val improvement for {epochs_no_improve} evals)"
                )
                break

    logger.info(
        f"[GCNAdvanced] Done. Best val@20={best_valid:.4f} | test@20={best_test:.4f} "
        f"(epoch {best_epoch})"
    )

    return GCNAdvancedResult(best_valid, best_test, best_epoch)


def main():
    parser = argparse.ArgumentParser(
        description='Unified training script for link prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=ALL_MODELS,
                        help='Model to train')

    # External features
    parser.add_argument('--external-features', type=str, default=None,
                        help='External features to use: "all", "none", or comma-separated list '
                             '(e.g., "morgan,chemberta"). Options: morgan, pubchem, chemberta, drug-targets')

    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--hidden-dim', type=int, default=128,
                             help='Hidden dimension (default: 128)')
    model_group.add_argument('--num-layers', type=int, default=2,
                             help='Number of GNN layers (default: 2)')
    model_group.add_argument('--dropout', type=float, default=0.0,
                             help='Dropout rate (default: 0.0)')
    model_group.add_argument('--decoder-dropout', type=float, default=0.0,
                             help='Decoder dropout rate (default: 0.0)')
    model_group.add_argument('--attention-heads', type=int, default=4,
                             help='Number of attention heads for GAT/Transformer (default: 4)')
    model_group.add_argument('--fusion', type=str, default='concat',
                             choices=['concat', 'add'],
                             help='Feature fusion strategy (default: concat)')

    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs', type=int, default=200,
                             help='Maximum training epochs (default: 200)')
    train_group.add_argument('--lr', type=float, default=0.01,
                             help='Learning rate (default: 0.01)')
    train_group.add_argument('--weight-decay', type=float, default=1e-4,
                             help='Weight decay (default: 1e-4)')
    train_group.add_argument('--batch-size', type=int, default=50000,
                             help='Training batch size (default: 50000)')
    train_group.add_argument('--eval-batch-size', type=int, default=50000,
                             help='Evaluation batch size (default: 50000)')
    train_group.add_argument('--num-neg', type=int, default=3,
                             help='Negatives per positive for GDIN (default: 3)')
    train_group.add_argument('--patience', type=int, default=20,
                             help='Early stopping patience (default: 20)')
    train_group.add_argument('--eval-every', type=int, default=5,
                             help='Evaluate every N epochs (default: 5)')

    # System configuration
    sys_group = parser.add_argument_group('System Configuration')
    sys_group.add_argument('--device', type=str, default=None,
                          choices=['cpu', 'cuda', 'mps'],
                          help='Device to use (default: auto-detect)')
    sys_group.add_argument('--data-dir', type=Path, default=Path('data'),
                          help='Directory for data files (default: data)')
    sys_group.add_argument('--seed', type=int, default=42,
                          help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup logging
    log_dir, log_file = setup_logging(args)
    logger = logging.getLogger(__name__)

    # Get device
    device = get_device(args)

    logger.info("=" * 80)
    logger.info(f"Training {args.model.upper()} Model")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Random seed: {args.seed}")

    # Load dataset
    logger.info("Loading OGBL-DDI dataset...")
    data, split_edge, num_nodes, evaluator = load_dataset('ogbl-ddi', device=device)

    train_pos = split_edge['train']['edge'].to(device)
    valid_pos = split_edge['valid']['edge'].to(device)
    valid_neg = split_edge['valid']['edge_neg'].to(device)
    test_pos = split_edge['test']['edge'].to(device)
    test_neg = split_edge['test']['edge_neg'].to(device)

    logger.info(f"Dataset: {num_nodes} nodes")
    logger.info(f"Train edges: {train_pos.size(0)}")
    logger.info(f"Valid edges: {valid_pos.size(0)} pos, {valid_neg.size(0)} neg")
    logger.info(f"Test edges: {test_pos.size(0)} pos, {test_neg.size(0)} neg")

    # Load external features if needed
    external_features, external_dims = load_external_features_if_needed(
        args, num_nodes, logger
    )

    # Extract specific feature types for chemistry models
    chemberta_features = None
    morgan_features = None
    if external_features:
        if 'chemberta' in external_features:
            chemberta_features = external_features['chemberta'].to(device)
            logger.info(f"ChemBERTa features: {chemberta_features.shape}")
        if 'morgan' in external_features:
            morgan_features = external_features['morgan'].to(device)
            logger.info(f"Morgan features: {morgan_features.shape}")

    # Create and train model
    splits = (train_pos, valid_pos, valid_neg, test_pos, test_neg)

    if args.model == 'gdinn':
        # GDINN requires external features
        model = create_model(args, num_nodes, external_dims, chemberta_features, morgan_features, device, logger)
        result = train_gdinn_model(args, model, data, splits, external_features, device, logger)
    elif args.model == 'gcn-advanced':
        # GCNAdvanced uses OGB reference training
        model = create_model(args, num_nodes, external_dims, chemberta_features, morgan_features, device, logger)
        result = train_gcn_advanced_model(args, model, data, splits, device, logger)
    else:
        # Baseline, chemistry, and hybrid models use the minimal trainer
        model = create_model(args, num_nodes, external_dims, chemberta_features, morgan_features, device, logger)
        result = train_baseline_model(args, model, data, splits, chemberta_features, morgan_features, device, logger)

    # Final summary
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best Validation Hits@20: {result.best_val_hits:.4f}")
    logger.info(f"Best Test Hits@20: {result.best_test_hits:.4f}")
    logger.info(f"Best Epoch: {result.best_epoch}")
    logger.info(f"Results logged to: {log_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

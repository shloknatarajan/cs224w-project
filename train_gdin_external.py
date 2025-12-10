#!/usr/bin/env python
"""
Train GDIN with External Knowledge Features

This script trains the GDINMultiModal model with configurable external
knowledge sources:

- Phase 1: Morgan fingerprints (molecular substructure)
- Phase 2: PubChem properties (physicochemical features)
- Phase 3: ChemBERTa embeddings (pre-trained molecular representations)
- Phase 4: Drug-target interactions (biological context)

Usage:
    # Train with all features
    python train_gdin_external.py --all

    # Train with specific features
    python train_gdin_external.py --morgan --pubchem

    # Train with ChemBERTa only
    python train_gdin_external.py --chemberta

    # Enable common neighbor features (from original GDIN)
    python train_gdin_external.py --all --use-cn
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import scipy.sparse as sp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import load_dataset
from src.data.external_features import (
    FeatureConfig,
    load_external_features,
    get_default_config,
    get_minimal_config,
)
from src.models.advanced import GDINMultiModal, create_gdin_multimodal
from src.training.gdin_trainer import train_gdin_multimodal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_adjacency_sparse(edge_index: torch.Tensor, num_nodes: int):
    """Build scipy sparse adjacency matrix for CN computation."""
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = [1] * len(row)
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj


def main():
    parser = argparse.ArgumentParser(
        description="Train GDIN with external knowledge features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Feature selection
    feature_group = parser.add_argument_group("Feature Selection")
    feature_group.add_argument("--all", action="store_true",
                               help="Enable all available features")
    feature_group.add_argument("--morgan", action="store_true",
                               help="Use Morgan fingerprints (Phase 1)")
    feature_group.add_argument("--pubchem", action="store_true",
                               help="Use PubChem properties (Phase 2)")
    feature_group.add_argument("--chemberta", action="store_true",
                               help="Use ChemBERTa embeddings (Phase 3)")
    feature_group.add_argument("--drug-targets", action="store_true",
                               help="Use drug-target features (Phase 4)")

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--hidden-dim", type=int, default=256,
                             help="Hidden dimension (default: 256)")
    model_group.add_argument("--num-layers", type=int, default=2,
                             help="Number of GNN layers (default: 2)")
    model_group.add_argument("--dropout", type=float, default=0.1,
                             help="Dropout rate (default: 0.1)")
    model_group.add_argument("--fusion", type=str, default="attention",
                             choices=["attention", "gated", "concat"],
                             help="Feature fusion strategy (default: attention)")
    model_group.add_argument("--use-cn", action="store_true",
                             help="Enable common neighbor features")

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--epochs", type=int, default=400,
                             help="Max training epochs (default: 400)")
    train_group.add_argument("--lr", type=float, default=0.005,
                             help="Learning rate (default: 0.005)")
    train_group.add_argument("--weight-decay", type=float, default=1e-4,
                             help="Weight decay (default: 1e-4)")
    train_group.add_argument("--batch-size", type=int, default=50000,
                             help="Training batch size (default: 50000)")
    train_group.add_argument("--num-neg", type=int, default=3,
                             help="Negatives per positive (default: 3)")
    train_group.add_argument("--patience", type=int, default=30,
                             help="Early stopping patience (default: 30)")
    train_group.add_argument("--eval-every", type=int, default=5,
                             help="Evaluate every N epochs (default: 5)")

    # Data paths
    path_group = parser.add_argument_group("Data Paths")
    path_group.add_argument("--data-dir", type=Path, default=Path("data"),
                            help="Directory for external feature files")
    path_group.add_argument("--smiles-csv", type=Path, default=None,
                            help="Path to SMILES CSV (for Morgan features)")

    args = parser.parse_args()

    # Determine which features to use
    if args.all:
        use_morgan = True
        use_pubchem = True
        use_chemberta = True
        use_drug_targets = True
    else:
        use_morgan = args.morgan
        use_pubchem = args.pubchem
        use_chemberta = args.chemberta
        use_drug_targets = args.drug_targets

    # If no features specified, default to Morgan only
    if not any([use_morgan, use_pubchem, use_chemberta, use_drug_targets]):
        logger.warning("No features specified. Using Morgan fingerprints by default.")
        use_morgan = True

    # Create feature config
    feature_config = FeatureConfig(
        use_morgan=use_morgan,
        use_pubchem=use_pubchem,
        use_chemberta=use_chemberta,
        use_drug_targets=use_drug_targets,
        morgan_path=args.data_dir / "morgan_features.pt",
        pubchem_path=args.data_dir / "ogbl_ddi_properties.csv",
        chemberta_path=args.data_dir / "chemberta_embeddings.pt",
        drug_targets_path=args.data_dir / "drug_target_features.pt",
    )

    logger.info("=" * 60)
    logger.info("GDIN with External Knowledge Training")
    logger.info("=" * 60)
    logger.info(f"Features enabled:")
    logger.info(f"  - Morgan fingerprints: {use_morgan}")
    logger.info(f"  - PubChem properties: {use_pubchem}")
    logger.info(f"  - ChemBERTa embeddings: {use_chemberta}")
    logger.info(f"  - Drug-target features: {use_drug_targets}")
    logger.info(f"  - Common neighbors (CN): {args.use_cn}")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading OGBL-DDI dataset...")
    smiles_path = args.smiles_csv or args.data_dir / "ogbl_ddi_smiles.csv"
    data, split_edge, num_nodes, evaluator = load_dataset(
        'ogbl-ddi',
        device=str(device),
        smiles_csv_path=str(smiles_path) if smiles_path.exists() else None,
    )

    # Get edge splits
    train_pos = split_edge['train']['edge'].to(device)
    valid_pos = split_edge['valid']['edge'].to(device)
    valid_neg = split_edge['valid']['edge_neg'].to(device)
    test_pos = split_edge['test']['edge'].to(device)
    test_neg = split_edge['test']['edge_neg'].to(device)

    logger.info(f"Dataset: {num_nodes} nodes")
    logger.info(f"Train edges: {train_pos.size(0)}")
    logger.info(f"Valid edges: {valid_pos.size(0)} pos, {valid_neg.size(0)} neg")
    logger.info(f"Test edges: {test_pos.size(0)} pos, {test_neg.size(0)} neg")

    # Load external features
    logger.info("Loading external features...")
    ext_features = load_external_features(
        feature_config, num_nodes,
        smiles_csv_path=smiles_path if smiles_path.exists() else None,
    )

    if ext_features.total_dim == 0:
        logger.error("No external features could be loaded!")
        logger.error("Run the data fetching scripts first:")
        logger.error("  python -m src.data.fetch_smiles")
        logger.error("  python -m src.data.fetch_pubchem_properties")
        logger.error("  python -m src.data.extract_chemberta_embeddings")
        logger.error("  python -m src.data.fetch_drug_targets")
        sys.exit(1)

    logger.info(f"Loaded features: {ext_features.feature_dims}")
    logger.info(f"Total feature dimension: {ext_features.total_dim}")

    # Build sparse adjacency for CN computation
    adj_sparse = None
    if args.use_cn:
        logger.info("Building sparse adjacency matrix for CN computation...")
        adj_sparse = build_adjacency_sparse(data.edge_index, num_nodes)

    # Create model
    logger.info("Creating GDINMultiModal model...")
    model = create_gdin_multimodal(
        num_nodes=num_nodes,
        feature_dims=ext_features.feature_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        fusion=args.fusion,
        use_cn=args.use_cn,
    )
    logger.info(f"Model: {model.description}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare external features dict for training
    # Map from our feature names to what the model expects
    external_features_dict = {}
    if ext_features.morgan is not None:
        external_features_dict['morgan'] = ext_features.morgan
    if ext_features.pubchem is not None:
        external_features_dict['pubchem'] = ext_features.pubchem
    if ext_features.chemberta is not None:
        external_features_dict['chemberta'] = ext_features.chemberta
    if ext_features.drug_targets is not None:
        external_features_dict['drug_targets'] = ext_features.drug_targets

    # Train model
    logger.info("Starting training...")
    result = train_gdin_multimodal(
        name="GDINMultiModal",
        model=model,
        data=data,
        train_pos=train_pos,
        valid_pos=valid_pos,
        valid_neg=valid_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        external_features=external_features_dict,
        evaluator=evaluator,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_neg=args.num_neg,
        patience=args.patience,
        eval_every=args.eval_every,
        use_cn=args.use_cn,
        adj_sparse=adj_sparse,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Validation Hits@20: {result.best_val_hits:.4f}")
    logger.info(f"Best Test Hits@20: {result.best_test_hits:.4f}")
    logger.info(f"Best Epoch: {result.best_epoch}")


if __name__ == "__main__":
    main()

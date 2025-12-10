"""
Fetch drug-target interaction data for OGBL-DDI drugs.

Uses Therapeutics Data Commons (TDC) to get drug-target interactions,
which can help DDI prediction by providing biological context.

The OGBL-DDI dataset uses protein-target based splits, so drug-target
knowledge is particularly valuable for out-of-distribution generalization.
"""
import gzip
import csv
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MAPPING_PATH = Path.home() / "cs224w-project/dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
DEFAULT_SMILES_PATH = Path(__file__).parent.parent.parent / "data/ogbl_ddi_smiles.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent / "data/drug_target_features.pt"
DEFAULT_EDGES_PATH = Path(__file__).parent.parent.parent / "data/drug_target_edges.pt"


def load_tdc_drug_targets() -> pd.DataFrame:
    """
    Load drug-target interaction data from TDC.

    Returns DataFrame with columns: Drug_ID, Drug, Target_ID, Target, Y
    """
    # Import TDC here to avoid loading unless needed
    from tdc.multi_pred import DTI

    logger.info("Loading drug-target data from TDC (KIBA dataset)...")
    dti = DTI(name='KIBA')
    df = dti.get_data()

    logger.info(f"Loaded {len(df)} drug-target interactions")
    logger.info(f"Unique drugs: {df['Drug_ID'].nunique()}, Unique targets: {df['Target_ID'].nunique()}")

    return df


def build_drug_smiles_to_target_map(dti_df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Build mapping from drug SMILES to set of target IDs.

    TDC uses SMILES as drug identifiers, so we match by SMILES.
    """
    drug_to_targets = defaultdict(set)

    for _, row in dti_df.iterrows():
        drug_smiles = row['Drug']  # In TDC, 'Drug' column contains SMILES
        target_id = row['Target_ID']
        drug_to_targets[drug_smiles].add(target_id)

    return dict(drug_to_targets)


def fetch_drug_target_features(
    smiles_path: Path = DEFAULT_SMILES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    edges_path: Path = DEFAULT_EDGES_PATH,
    max_targets: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create drug-target feature vectors and edge list for OGBL-DDI drugs.

    Args:
        smiles_path: Path to SMILES CSV (from fetch_smiles.py)
        output_path: Where to save the target feature tensor
        edges_path: Where to save drug-target edges
        max_targets: Maximum number of target dimensions (for PCA if needed)

    Returns:
        Tuple of:
        - target_features: [num_drugs, num_targets] binary tensor
        - drug_target_edges: [2, num_edges] edge tensor for heterogeneous graph
    """
    # Load our SMILES data
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"SMILES file not found at {smiles_path}. "
            "Run fetch_smiles.py first."
        )

    smiles_df = pd.read_csv(smiles_path)
    num_drugs = len(smiles_df)
    logger.info(f"Loaded {num_drugs} drugs from SMILES file")

    # Load TDC data
    dti_df = load_tdc_drug_targets()

    # Build SMILES to target mapping
    drug_to_targets = build_drug_smiles_to_target_map(dti_df)

    # Get all unique targets
    all_targets = set()
    for targets in drug_to_targets.values():
        all_targets.update(targets)
    all_targets = sorted(all_targets)

    # Limit to top N targets by frequency if needed
    if len(all_targets) > max_targets:
        target_counts = defaultdict(int)
        for targets in drug_to_targets.values():
            for t in targets:
                target_counts[t] += 1
        top_targets = sorted(target_counts.keys(), key=lambda x: -target_counts[x])[:max_targets]
        all_targets = top_targets
        logger.info(f"Limited to top {max_targets} targets by frequency")

    target_to_idx = {t: i for i, t in enumerate(all_targets)}
    num_targets = len(all_targets)
    logger.info(f"Using {num_targets} unique targets")

    # Build feature matrix and edge list
    target_features = torch.zeros(num_drugs, num_targets)
    edges = []  # (drug_idx, target_idx) pairs
    matched_drugs = 0

    for _, row in smiles_df.iterrows():
        drug_idx = row['ogb_id']
        drug_smiles = row['smiles']

        if pd.isna(drug_smiles):
            continue

        # Try to match by SMILES
        if drug_smiles in drug_to_targets:
            targets = drug_to_targets[drug_smiles]
            matched_drugs += 1

            for target in targets:
                if target in target_to_idx:
                    target_idx = target_to_idx[target]
                    target_features[drug_idx, target_idx] = 1.0
                    edges.append((drug_idx, target_idx))

    logger.info(f"Matched {matched_drugs}/{num_drugs} drugs to TDC data")
    logger.info(f"Total drug-target edges: {len(edges)}")

    # Create edge tensor
    if edges:
        drug_target_edges = torch.tensor(edges, dtype=torch.long).t()
    else:
        drug_target_edges = torch.zeros(2, 0, dtype=torch.long)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target_features, output_path)
    torch.save({
        'edges': drug_target_edges,
        'num_drugs': num_drugs,
        'num_targets': num_targets,
        'target_to_idx': target_to_idx,
    }, edges_path)

    logger.info(f"Saved target features to {output_path}")
    logger.info(f"Saved drug-target edges to {edges_path}")

    return target_features, drug_target_edges


def load_drug_target_features(path: Path = DEFAULT_OUTPUT_PATH) -> torch.Tensor:
    """Load pre-computed drug-target feature vectors."""
    if not path.exists():
        raise FileNotFoundError(
            f"Drug-target features not found at {path}. "
            "Run fetch_drug_target_features() first."
        )
    return torch.load(path, weights_only=True)


def load_drug_target_edges(path: Path = DEFAULT_EDGES_PATH) -> dict:
    """Load pre-computed drug-target edges for heterogeneous graph."""
    if not path.exists():
        raise FileNotFoundError(
            f"Drug-target edges not found at {path}. "
            "Run fetch_drug_target_features() first."
        )
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch drug-target data for OGBL-DDI drugs")
    parser.add_argument("--smiles", type=Path, default=DEFAULT_SMILES_PATH,
                        help="Path to SMILES CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="Output tensor path for features (.pt)")
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES_PATH,
                        help="Output path for edges (.pt)")
    parser.add_argument("--max-targets", type=int, default=500,
                        help="Maximum number of target dimensions")

    args = parser.parse_args()

    fetch_drug_target_features(
        smiles_path=args.smiles,
        output_path=args.output,
        edges_path=args.edges,
        max_targets=args.max_targets,
    )

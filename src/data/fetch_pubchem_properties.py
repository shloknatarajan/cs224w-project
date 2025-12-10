"""
Fetch molecular properties from PubChem for OGBL-DDI drugs.

Fetches pharmacologically relevant properties:
- MolecularWeight, XLogP (lipophilicity), TPSA (polar surface area)
- HBondDonorCount, HBondAcceptorCount, RotatableBondCount
- HeavyAtomCount, Complexity, Charge
"""
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pubchempy as pcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SMILES_PATH = Path(__file__).parent.parent.parent / "data/ogbl_ddi_smiles.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent / "data/ogbl_ddi_properties.csv"

# Properties to fetch from PubChem
PROPERTY_LIST = [
    'MolecularWeight',
    'XLogP',              # Lipophilicity (octanol-water partition coefficient)
    'TPSA',               # Topological Polar Surface Area
    'HBondDonorCount',    # Hydrogen bond donors
    'HBondAcceptorCount', # Hydrogen bond acceptors
    'RotatableBondCount', # Rotatable bonds (flexibility)
    'HeavyAtomCount',     # Non-hydrogen atoms
    'Complexity',         # Molecular complexity score
    'Charge',             # Formal charge
]


def fetch_properties_by_smiles(smiles: str) -> dict:
    """
    Fetch molecular properties from PubChem by SMILES.

    Returns dict with property values (NaN for missing).
    """
    props = {p: np.nan for p in PROPERTY_LIST}

    if not smiles or pd.isna(smiles):
        return props

    try:
        compounds = pcp.get_compounds(smiles, 'smiles')
        if compounds:
            c = compounds[0]
            for prop in PROPERTY_LIST:
                # PubChem uses lowercase attribute names
                attr_name = prop.lower().replace('_', '')

                # Handle special cases
                if prop == 'MolecularWeight':
                    attr_name = 'molecular_weight'
                elif prop == 'XLogP':
                    attr_name = 'xlogp'
                elif prop == 'TPSA':
                    attr_name = 'tpsa'
                elif prop == 'HBondDonorCount':
                    attr_name = 'h_bond_donor_count'
                elif prop == 'HBondAcceptorCount':
                    attr_name = 'h_bond_acceptor_count'
                elif prop == 'RotatableBondCount':
                    attr_name = 'rotatable_bond_count'
                elif prop == 'HeavyAtomCount':
                    attr_name = 'heavy_atom_count'
                elif prop == 'Complexity':
                    attr_name = 'complexity'
                elif prop == 'Charge':
                    attr_name = 'charge'

                if hasattr(c, attr_name):
                    val = getattr(c, attr_name)
                    if val is not None:
                        props[prop] = float(val)

    except Exception as e:
        logger.debug(f"PubChem error for SMILES {smiles[:50]}...: {e}")

    return props


def fetch_all_properties(
    smiles_path: Path = DEFAULT_SMILES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    rate_limit: float = 0.2,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Fetch molecular properties for all drugs with SMILES.

    Args:
        smiles_path: Path to SMILES CSV (from fetch_smiles.py)
        output_path: Where to save the results
        rate_limit: Seconds to wait between API calls
        resume: If True, skip drugs already in output file

    Returns:
        DataFrame with columns: ogb_id + all PROPERTY_LIST
    """
    # Load SMILES data
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"SMILES file not found at {smiles_path}. "
            "Run fetch_smiles.py first."
        )

    smiles_df = pd.read_csv(smiles_path)
    num_drugs = len(smiles_df)
    has_smiles = smiles_df['smiles'].notna().sum()
    logger.info(f"Loaded {num_drugs} drugs, {has_smiles} have SMILES")

    # Check for existing results to resume
    existing_ids = set()
    if resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_ids = set(existing_df['ogb_id'].tolist())
        logger.info(f"Resuming: {len(existing_ids)} drugs already fetched")

    # Fetch properties for each drug
    results = []

    for idx, row in smiles_df.iterrows():
        ogb_id = row['ogb_id']

        if ogb_id in existing_ids:
            continue

        smiles = row['smiles']
        props = fetch_properties_by_smiles(smiles)
        props['ogb_id'] = ogb_id

        if pd.notna(smiles):
            time.sleep(rate_limit)  # Rate limiting only for API calls

        results.append(props)

        if (idx + 1) % 100 == 0:
            valid_count = sum(
                1 for r in results
                if not np.isnan(r.get('MolecularWeight', np.nan))
            )
            logger.info(f"Progress: {idx + 1}/{num_drugs} drugs, {valid_count} have properties")
            # Save intermediate results
            _save_results(results, existing_ids, output_path)

    # Save final results
    df = _save_results(results, existing_ids, output_path)

    valid_count = df['MolecularWeight'].notna().sum()
    logger.info(f"Done! {valid_count}/{num_drugs} drugs have properties ({100*valid_count/num_drugs:.1f}%)")

    return df


def _save_results(
    new_results: list[dict],
    existing_ids: set,
    output_path: Path,
) -> pd.DataFrame:
    """Save results, merging with existing file if present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if existing_ids and output_path.exists():
        existing_df = pd.read_csv(output_path)
        new_df = pd.DataFrame(new_results)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(new_results)

    # Reorder columns
    cols = ['ogb_id'] + PROPERTY_LIST
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values('ogb_id').drop_duplicates(subset='ogb_id', keep='last')
    df.to_csv(output_path, index=False)

    return df


def load_properties_csv(path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Load the properties CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Properties file not found at {path}. "
            "Run fetch_all_properties() first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch molecular properties for OGBL-DDI drugs")
    parser.add_argument("--smiles", type=Path, default=DEFAULT_SMILES_PATH,
                        help="Path to SMILES CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="Output CSV path")
    parser.add_argument("--rate-limit", type=float, default=0.2,
                        help="Seconds between API calls")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume from existing file")

    args = parser.parse_args()

    fetch_all_properties(
        smiles_path=args.smiles,
        output_path=args.output,
        rate_limit=args.rate_limit,
        resume=not args.no_resume,
    )

"""
Fetch SMILES strings for OGBL-DDI drugs from PubChem.

Uses the nodeidx2drugid.csv.gz mapping file and ddi_description.csv.gz
for drug names, then queries PubChem to get SMILES.
"""
import gzip
import csv
import time
import logging
from pathlib import Path

import pandas as pd
import pubchempy as pcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MAPPING_PATH = Path.home() / "cs224w-project/dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
DEFAULT_DESCRIPTION_PATH = Path.home() / "cs224w-project/dataset/ogbl_ddi/mapping/ddi_description.csv.gz"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent / "data/ogbl_ddi_smiles.csv"


def load_drug_mapping(mapping_path: Path) -> dict[int, str]:
    """Load node index to DrugBank ID mapping."""
    with gzip.open(mapping_path, 'rt') as f:
        reader = csv.DictReader(f)
        return {int(row['node idx']): row['drug id'] for row in reader}


def load_drug_names(description_path: Path) -> dict[str, str]:
    """Load DrugBank ID to drug name mapping from DDI descriptions."""
    drug_names = {}
    with gzip.open(description_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_names[row['first drug id']] = row['first drug name']
            drug_names[row['second drug id']] = row['second drug name']
    return drug_names


def fetch_smiles_from_pubchem(drug_name: str, drugbank_id: str) -> str | None:
    """
    Fetch SMILES from PubChem by drug name.

    Returns canonical SMILES or None if not found.
    """
    try:
        # Try searching by drug name first
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            return compounds[0].canonical_smiles

        # Fallback: try DrugBank ID as synonym
        compounds = pcp.get_compounds(drugbank_id, 'name')
        if compounds:
            return compounds[0].canonical_smiles

    except Exception as e:
        logger.debug(f"PubChem error for {drug_name} ({drugbank_id}): {e}")

    return None


def fetch_all_smiles(
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    description_path: Path = DEFAULT_DESCRIPTION_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    rate_limit: float = 0.2,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Fetch SMILES for all drugs in the OGBL-DDI dataset.

    Args:
        mapping_path: Path to nodeidx2drugid.csv.gz
        description_path: Path to ddi_description.csv.gz
        output_path: Where to save the results
        rate_limit: Seconds to wait between API calls
        resume: If True, skip drugs already in output file

    Returns:
        DataFrame with columns: ogb_id, drugbank_id, drug_name, smiles
    """
    # Load mappings
    logger.info(f"Loading drug mappings from {mapping_path}")
    node_to_drugbank = load_drug_mapping(mapping_path)

    logger.info(f"Loading drug names from {description_path}")
    drugbank_to_name = load_drug_names(description_path)

    num_nodes = len(node_to_drugbank)
    logger.info(f"Found {num_nodes} drugs, {len(drugbank_to_name)} have names")

    # Check for existing results to resume
    existing_ids = set()
    if resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_ids = set(existing_df['ogb_id'].tolist())
        logger.info(f"Resuming: {len(existing_ids)} drugs already fetched")

    # Fetch SMILES for each drug
    results = []
    success_count = 0

    for node_idx in range(num_nodes):
        if node_idx in existing_ids:
            continue

        drugbank_id = node_to_drugbank[node_idx]
        drug_name = drugbank_to_name.get(drugbank_id, "")

        smiles = None
        if drug_name:
            smiles = fetch_smiles_from_pubchem(drug_name, drugbank_id)
            time.sleep(rate_limit)  # Rate limiting

        if smiles:
            success_count += 1

        results.append({
            'ogb_id': node_idx,
            'drugbank_id': drugbank_id,
            'drug_name': drug_name,
            'smiles': smiles
        })

        if (node_idx + 1) % 100 == 0:
            logger.info(f"Progress: {node_idx + 1}/{num_nodes} drugs, {success_count} SMILES found")
            # Save intermediate results
            _save_results(results, existing_ids, output_path, mapping_path, description_path)

    # Save final results
    df = _save_results(results, existing_ids, output_path, mapping_path, description_path)

    total_smiles = df['smiles'].notna().sum()
    logger.info(f"Done! {total_smiles}/{num_nodes} drugs have SMILES ({100*total_smiles/num_nodes:.1f}%)")

    return df


def _save_results(
    new_results: list[dict],
    existing_ids: set,
    output_path: Path,
    mapping_path: Path,
    description_path: Path,
) -> pd.DataFrame:
    """Save results, merging with existing file if present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if existing_ids and output_path.exists():
        existing_df = pd.read_csv(output_path)
        new_df = pd.DataFrame(new_results)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(new_results)

    df = df.sort_values('ogb_id').drop_duplicates(subset='ogb_id', keep='last')
    df.to_csv(output_path, index=False)

    return df


def load_smiles_csv(path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Load the SMILES CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"SMILES file not found at {path}. "
            "Run fetch_all_smiles() first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch SMILES for OGBL-DDI drugs")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH,
                        help="Path to nodeidx2drugid.csv.gz")
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTION_PATH,
                        help="Path to ddi_description.csv.gz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="Output CSV path")
    parser.add_argument("--rate-limit", type=float, default=0.2,
                        help="Seconds between API calls")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume from existing file")

    args = parser.parse_args()

    fetch_all_smiles(
        mapping_path=args.mapping,
        description_path=args.descriptions,
        output_path=args.output,
        rate_limit=args.rate_limit,
        resume=not args.no_resume,
    )

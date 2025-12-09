"""
Fast SMILES fetcher using batch requests and caching.

This version uses:
1. Batch requests where possible
2. Parallel processing
3. Multiple fallback sources
"""
import pandas as pd
import gzip
import requests
import time
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_single_drug(node_id: int, drug_id: str) -> Tuple[int, str, str]:
    """
    Fetch SMILES for a single drug using multiple sources.

    Returns:
        (node_id, drug_id, smiles or '')
    """
    # Try Method 1: PubChem via DrugBank ID as synonym
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/synonym/{drug_id}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                smiles = data['PropertyTable']['Properties'][0].get('CanonicalSMILES', '')
                if smiles:
                    return (node_id, drug_id, smiles)
    except:
        pass

    # Try Method 2: ChEMBL via UniChem
    try:
        # Map DrugBank (source 2) to ChEMBL (source 1)
        url = f"https://www.ebi.ac.uk/unichem/rest/src_compound_id/{drug_id}/2/1"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                chembl_id = data[0]['src_compound_id']

                # Get SMILES from ChEMBL
                url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if 'molecule_structures' in data and data['molecule_structures']:
                        smiles = data['molecule_structures'].get('canonical_smiles', '')
                        if smiles:
                            return (node_id, drug_id, smiles)
    except:
        pass

    return (node_id, drug_id, '')


def main():
    # Read node to DrugBank ID mapping
    mapping_path = "dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    logger.info(f"Reading node-to-drug mapping from {mapping_path}")

    with gzip.open(mapping_path, 'rt') as f:
        df = pd.read_csv(f)

    logger.info(f"Found {len(df)} nodes/drugs")
    logger.info("Starting parallel fetch with 10 workers...")

    results = []
    failed_count = 0
    success_count = 0

    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = {
            executor.submit(fetch_single_drug, row['node idx'], row['drug id']): row
            for _, row in df.iterrows()
        }

        # Process completed tasks
        for i, future in enumerate(as_completed(futures)):
            node_id, drug_id, smiles = future.result()

            if smiles:
                success_count += 1
                logger.debug(f"✓ {drug_id}: {smiles[:50]}")
            else:
                failed_count += 1
                logger.debug(f"✗ {drug_id}: No SMILES found")

            results.append({'ogb_id': node_id, 'smiles': smiles})

            # Progress update every 100 drugs
            if (i + 1) % 100 == 0:
                pct_complete = 100 * (i + 1) / len(df)
                success_rate = 100 * success_count / (i + 1)
                logger.info(
                    f"Progress: {i+1}/{len(df)} ({pct_complete:.1f}%) | "
                    f"Success: {success_count} ({success_rate:.1f}%) | "
                    f"Failed: {failed_count}"
                )

    # Sort by node ID
    results_df = pd.DataFrame(results).sort_values('ogb_id').reset_index(drop=True)

    # Save results
    output_path = "data/smiles.csv"
    Path("data").mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Saved SMILES data to {output_path}")
    logger.info(f"✓ Successfully fetched: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    logger.info(f"✗ Failed to fetch: {failed_count}")
    logger.info(f"{'='*80}")

    if failed_count > 0:
        failed_ids = [row['drug id'] for _, row in df.iterrows()
                     if results_df[results_df['ogb_id'] == row['node idx']]['smiles'].values[0] == '']
        failed_path = "data/failed_drugbank_ids.txt"
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_ids))
        logger.info(f"Failed DrugBank IDs saved to {failed_path}")


if __name__ == "__main__":
    main()

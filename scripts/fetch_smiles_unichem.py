"""
Fetch SMILES strings for DrugBank IDs using multiple sources.

This script uses UniChem API to map DrugBank IDs to ChEMBL IDs, then fetches SMILES.
"""
import pandas as pd
import gzip
import requests
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_smiles_via_unichem(drugbank_id: str) -> str | None:
    """
    Fetch SMILES for a DrugBank ID via UniChem -> ChEMBL.

    Args:
        drugbank_id: DrugBank ID (e.g., 'DB00001')

    Returns:
        SMILES string or None if not found
    """
    try:
        # UniChem source IDs: DrugBank=2, ChEMBL=1
        # Map DrugBank to ChEMBL
        unichem_url = f"https://www.ebi.ac.uk/unichem/rest/src_compound_id/{drugbank_id}/2/1"
        response = requests.get(unichem_url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data or len(data) == 0:
            return None

        chembl_id = data[0]['src_compound_id']

        # Get SMILES from ChEMBL
        chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        response = requests.get(chembl_url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if 'molecule_structures' in data and data['molecule_structures']:
            smiles = data['molecule_structures'].get('canonical_smiles')
            return smiles

    except Exception as e:
        logger.debug(f"Error fetching SMILES for {drugbank_id}: {e}")
        return None

    return None


def fetch_smiles_from_pubchem_direct(drugbank_id: str) -> str | None:
    """
    Try direct search on PubChem using compound name from DrugBank.
    """
    try:
        # Try searching by synonym (DrugBank ID)
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drugbank_id}/property/CanonicalSMILES/JSON"
        response = requests.get(search_url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                smiles = data['PropertyTable']['Properties'][0].get('CanonicalSMILES')
                return smiles

    except Exception as e:
        logger.debug(f"PubChem direct search failed for {drugbank_id}: {e}")

    return None


def main():
    # Read node to DrugBank ID mapping
    mapping_path = "dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    logger.info(f"Reading node-to-drug mapping from {mapping_path}")

    with gzip.open(mapping_path, 'rt') as f:
        df = pd.read_csv(f)

    logger.info(f"Found {len(df)} nodes/drugs")

    # Create output dataframe
    results = []
    failed_drugs = []
    success_count = 0

    # Fetch SMILES for each drug
    for idx, row in df.iterrows():
        node_id = row['node idx']
        drug_id = row['drug id']

        if idx % 100 == 0:
            logger.info(f"Progress: {idx}/{len(df)} ({100*idx/len(df):.1f}%) - Success rate: {100*success_count/(idx+1):.1f}%")

        logger.debug(f"Fetching SMILES for node {node_id}: {drug_id}")

        # Try UniChem first
        smiles = fetch_smiles_via_unichem(drug_id)

        if not smiles:
            # Try PubChem as fallback
            smiles = fetch_smiles_from_pubchem_direct(drug_id)

        if smiles:
            results.append({'ogb_id': node_id, 'smiles': smiles})
            success_count += 1
            logger.debug(f"  ✓ Found SMILES: {smiles[:50]}...")
        else:
            # Add empty string for missing SMILES
            results.append({'ogb_id': node_id, 'smiles': ''})
            failed_drugs.append(drug_id)
            logger.debug(f"  ✗ Could not fetch SMILES for {drug_id}")

        # Rate limiting
        time.sleep(0.15)

    # Save results
    output_path = "data/smiles.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    logger.info(f"\n{'='*80}")
    logger.info(f"Saved SMILES data to {output_path}")
    logger.info(f"Successfully fetched: {success_count}/{len(results)} drugs ({100*success_count/len(results):.1f}%)")
    logger.info(f"Failed to fetch: {len(failed_drugs)} drugs")

    if failed_drugs:
        logger.info(f"\nFailed DrugBank IDs (first 10): {', '.join(failed_drugs[:10])}")

        # Save failed IDs for reference
        failed_path = "data/failed_drugbank_ids.txt"
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_drugs))
        logger.info(f"Full list of failed IDs saved to {failed_path}")

    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

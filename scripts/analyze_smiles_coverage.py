"""
Analyze SMILES coverage in the ogbl-ddi dataset.

This script provides detailed statistics on:
1. Overall SMILES coverage
2. Coverage for nodes that appear in edges (train/valid/test)
3. Morgan fingerprint validity
4. Drug types (small molecules vs biotech)
"""
import pandas as pd
import gzip
import torch
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, '/home/ubuntu/cs224w-project')

from src.data.data_loader import smiles_to_morgan


def main():
    print("="*80)
    print("SMILES Coverage Analysis for ogbl-ddi Dataset")
    print("="*80)

    # Load DrugBank mapping
    print("\n1. Loading DrugBank mapping...")
    mapping_path = "dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    with gzip.open(mapping_path, 'rt') as f:
        mapping_df = pd.read_csv(f)

    total_nodes = len(mapping_df)
    print(f"   Total nodes in dataset: {total_nodes:,}")

    # Load SMILES data
    print("\n2. Loading SMILES data...")
    smiles_path = "data/smiles.csv"
    smiles_df = pd.read_csv(smiles_path, keep_default_na=False)

    # Analyze SMILES coverage
    non_empty_smiles = smiles_df[smiles_df['smiles'] != '']
    empty_smiles = smiles_df[smiles_df['smiles'] == '']

    num_with_smiles = len(non_empty_smiles)
    num_without_smiles = len(empty_smiles)
    coverage_pct = 100 * num_with_smiles / total_nodes

    print(f"   Nodes with SMILES: {num_with_smiles:,} ({coverage_pct:.1f}%)")
    print(f"   Nodes without SMILES: {num_without_smiles:,} ({100-coverage_pct:.1f}%)")

    # Analyze Morgan fingerprint validity
    print("\n3. Validating Morgan fingerprints...")
    valid_fps = 0
    invalid_fps = 0

    for idx, row in non_empty_smiles.iterrows():
        smiles = row['smiles']
        fp = smiles_to_morgan(smiles, n_bits=2048, radius=2)

        if torch.sum(fp) > 0:
            valid_fps += 1
        else:
            invalid_fps += 1

    print(f"   Valid Morgan FPs: {valid_fps:,} ({100*valid_fps/num_with_smiles:.1f}% of SMILES)")
    if invalid_fps > 0:
        print(f"   Invalid Morgan FPs: {invalid_fps:,} ({100*invalid_fps/num_with_smiles:.1f}% of SMILES)")

    # Load edge splits to see which nodes are actually used
    print("\n4. Analyzing nodes in edge splits...")

    # Load edge splits
    train_edges = torch.load("dataset/ogbl_ddi/split/target/train.pt")
    valid_edges = torch.load("dataset/ogbl_ddi/split/target/valid.pt")
    test_edges = torch.load("dataset/ogbl_ddi/split/target/test.pt")

    # Get unique nodes from edges
    train_nodes = set()
    train_nodes.update(train_edges['edge'][:, 0].tolist())
    train_nodes.update(train_edges['edge'][:, 1].tolist())

    valid_nodes = set()
    valid_nodes.update(valid_edges['edge'][:, 0].tolist())
    valid_nodes.update(valid_edges['edge'][:, 1].tolist())
    valid_nodes.update(valid_edges['edge_neg'][:, 0].tolist())
    valid_nodes.update(valid_edges['edge_neg'][:, 1].tolist())

    test_nodes = set()
    test_nodes.update(test_edges['edge'][:, 0].tolist())
    test_nodes.update(test_edges['edge'][:, 1].tolist())
    test_nodes.update(test_edges['edge_neg'][:, 0].tolist())
    test_nodes.update(test_edges['edge_neg'][:, 1].tolist())

    all_edge_nodes = train_nodes | valid_nodes | test_nodes

    print(f"   Nodes in train edges: {len(train_nodes):,}")
    print(f"   Nodes in valid edges: {len(valid_nodes):,}")
    print(f"   Nodes in test edges: {len(test_nodes):,}")
    print(f"   Total unique nodes in edges: {len(all_edge_nodes):,}")
    print(f"   Nodes never appearing in edges: {total_nodes - len(all_edge_nodes):,}")

    # Check SMILES coverage for nodes in edges
    print("\n5. SMILES coverage for nodes in edges...")

    edge_nodes_with_smiles = set()
    edge_nodes_without_smiles = set()

    for node_id in all_edge_nodes:
        smiles = smiles_df[smiles_df['ogb_id'] == node_id]['smiles'].values[0]
        if smiles and smiles != '':
            edge_nodes_with_smiles.add(node_id)
        else:
            edge_nodes_without_smiles.add(node_id)

    edge_coverage_pct = 100 * len(edge_nodes_with_smiles) / len(all_edge_nodes)

    print(f"   Nodes in edges WITH SMILES: {len(edge_nodes_with_smiles):,} ({edge_coverage_pct:.1f}%)")
    print(f"   Nodes in edges WITHOUT SMILES: {len(edge_nodes_without_smiles):,} ({100-edge_coverage_pct:.1f}%)")

    # Show some examples of drugs without SMILES
    print("\n6. Examples of drugs WITHOUT SMILES (biotech drugs):")
    missing_examples = empty_smiles.head(10)
    for idx, row in missing_examples.iterrows():
        node_id = row['ogb_id']
        drug_id = mapping_df[mapping_df['node idx'] == node_id]['drug id'].values[0]
        in_edges = "✓" if node_id in all_edge_nodes else "✗"
        print(f"   Node {node_id:4d} → {drug_id} [In edges: {in_edges}]")

    # Show some examples of drugs WITH SMILES
    print("\n7. Examples of drugs WITH SMILES (small molecules):")
    smiles_examples = non_empty_smiles.head(10)
    for idx, row in smiles_examples.iterrows():
        node_id = row['ogb_id']
        drug_id = mapping_df[mapping_df['node idx'] == node_id]['drug id'].values[0]
        smiles = row['smiles']
        in_edges = "✓" if node_id in all_edge_nodes else "✗"
        fp = smiles_to_morgan(smiles, n_bits=2048, radius=2)
        num_bits = torch.sum(fp).item()
        print(f"   Node {node_id:4d} → {drug_id} → {num_bits:3.0f} bits [In edges: {in_edges}]")
        print(f"          SMILES: {smiles[:60]}...")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total nodes in dataset:              {total_nodes:,}")
    print(f"Nodes with SMILES:                   {num_with_smiles:,} ({coverage_pct:.1f}%)")
    print(f"Nodes with valid Morgan FPs:         {valid_fps:,} ({100*valid_fps/total_nodes:.1f}%)")
    print(f"Nodes without SMILES:                {num_without_smiles:,} ({100-num_without_smiles/total_nodes:.1f}%)")
    print()
    print(f"Nodes appearing in edges:            {len(all_edge_nodes):,} ({100*len(all_edge_nodes)/total_nodes:.1f}%)")
    print(f"Edge nodes with SMILES:              {len(edge_nodes_with_smiles):,} ({edge_coverage_pct:.1f}%)")
    print(f"Edge nodes without SMILES:           {len(edge_nodes_without_smiles):,} ({100-edge_coverage_pct:.1f}%)")
    print()
    print("Impact on training:")
    print(f"  - {edge_coverage_pct:.1f}% of nodes in edges have molecular features")
    print(f"  - {100-edge_coverage_pct:.1f}% of nodes in edges have zero features (biotech drugs)")
    print("="*80)


if __name__ == "__main__":
    main()

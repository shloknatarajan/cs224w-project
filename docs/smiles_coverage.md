SMILES Coverage in ogbl-ddi Dataset
Overall Statistics

Total Dataset:
- Total nodes: 4,267 drugs
- Nodes with SMILES: 2,287 (53.6%)
- Nodes with valid Morgan fingerprints: 2,284 (53.5%)
- Nodes without SMILES: 1,980 (46.4%)

Key Finding: All Nodes Are Used in Training

Important: All 4,267 nodes appear in the edge splits
(train/valid/test), meaning:
- 100% of nodes are used in training
- 53.6% have molecular features (Morgan fingerprints)
- 46.4% have zero features (biotech drugs without
SMILES)

Coverage Details

Nodes with Valid Morgan Fingerprints:
- 2,284 out of 2,287 SMILES (99.9% success rate)
- Only 3 SMILES strings produced invalid fingerprints
(likely malformed)
- These represent small molecules (traditional drugs)

Nodes without SMILES:
- 1,980 drugs (46.4%)
- These are biotech drugs: proteins, peptides,
antibodies, large biologics
- Examples: DB00001 (Lepirudin), DB00002, DB00004,
etc.
- Cannot be represented with traditional SMILES
notation

Examples

Drugs WITH SMILES (small molecules):
- Node 4 (DB00006): 107 bits set - peptide/small
protein
- Node 11 (DB00014): 118 bits set - peptide
- Node 93 (DB00120): 22 bits set - Phenylalanine
(amino acid)

Drugs WITHOUT SMILES (biotech):
- Node 0 (DB00001): Lepirudin - recombinant protein
(65 amino acids)
- Nodes 1-3, 5-10: Various biotech drugs

Impact on Your Training

For the Morgan Baseline Models:
- ✅ 53.6% of nodes have real molecular features from
DrugBank SMILES
- ⚠️ 46.4% of nodes have zero-vector features (biotech
drugs)
- This is expected and correct - biotech drugs require
different representations

The models will learn from:
1. Molecular structure (Morgan fingerprints) for 53.6%
of drugs
2. Graph structure (edges/interactions) for all 100%
of drugs
3. Zero features for biotech drugs (which will rely
purely on graph connectivity)

This is actually a realistic scenario for drug-drug
interaction prediction, as many modern therapeutics
are biologics without traditional molecular
representations!
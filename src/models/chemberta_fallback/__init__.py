"""
ChemBERTa-based models with learnable fallback embeddings for missing SMILES.

These models use a hybrid approach:
- Nodes WITH SMILES: Use ChemBERTa embeddings
- Nodes WITHOUT SMILES: Use a learnable fallback embedding

This is better than using zeros for missing nodes.
"""
from .gcn import ChemBERTaGCNFallback
from .sage import ChemBERTaGraphSAGEFallback
from .gat import ChemBERTaGATFallback
from .transformer import ChemBERTaGraphTransformerFallback

__all__ = [
    'ChemBERTaGCNFallback',
    'ChemBERTaGraphSAGEFallback',
    'ChemBERTaGATFallback',
    'ChemBERTaGraphTransformerFallback',
]

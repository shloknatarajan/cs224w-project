"""
ChemBERTa-based baseline models.

These models use ChemBERTa embeddings (768-dim) as input features
instead of learnable embeddings or Morgan fingerprints.
"""
from .gcn import ChemBERTaGCN
from .sage import ChemBERTaGraphSAGE
from .gat import ChemBERTaGAT
from .transformer import ChemBERTaGraphTransformer

__all__ = [
    'ChemBERTaGCN',
    'ChemBERTaGraphSAGE',
    'ChemBERTaGAT',
    'ChemBERTaGraphTransformer',
]

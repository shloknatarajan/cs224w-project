"""
Node2Vec-enhanced GNN models for link prediction.

All models use Node2Vec embeddings as initialization, then fine-tune them
during link prediction training.
"""

from .gcn import Node2VecGCN
from .sage import Node2VecGraphSAGE
from .transformer import Node2VecTransformer
from .gat import Node2VecGAT

__all__ = [
    'Node2VecGCN',
    'Node2VecGraphSAGE',
    'Node2VecTransformer',
    'Node2VecGAT',
]


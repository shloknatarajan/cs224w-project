"""
Morgan fingerprint-based baseline models.

These models use Morgan fingerprints (2048-dim) as input features
instead of learnable embeddings.
"""
from .gcn import MorganGCN
from .sage import MorganGraphSAGE
from .gat import MorganGAT
from .transformer import MorganGraphTransformer

__all__ = [
    'MorganGCN',
    'MorganGraphSAGE',
    'MorganGAT',
    'MorganGraphTransformer',
]

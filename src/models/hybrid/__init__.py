"""
Hybrid Models: Structure-only encoder + Chemistry-aware decoder

These models combine the best of both worlds:
- Strong topological learning (like structure-only baselines)
- Intelligent use of chemistry (only when helpful)

Key innovation: Chemistry is incorporated at the decoder level, not the encoder.
This allows the model to learn when chemistry is useful vs when topology is better.
"""

from .decoder import ChemistryAwareDecoder, SimpleChemistryDecoder
from .gcn import HybridGCN
from .sage import HybridGraphSAGE
from .transformer import HybridGraphTransformer
from .gat import HybridGAT

__all__ = [
    'ChemistryAwareDecoder',
    'SimpleChemistryDecoder',
    'HybridGCN',
    'HybridGraphSAGE',
    'HybridGraphTransformer',
    'HybridGAT',
]


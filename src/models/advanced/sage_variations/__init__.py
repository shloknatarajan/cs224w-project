"""
GraphSAGE Architecture Variants for Testing

Quick architecture tests to find best configuration before hyperparameter sweeping.
All variants use lr=0.01 (same as baseline) for fair comparison.
"""

from .variant1_baseline import SAGEVariant1Baseline
from .variant2_batchnorm import SAGEVariant2BatchNorm
from .variant3_depth import SAGEVariant3Depth
from .variant4_both import SAGEVariant4Both

__all__ = [
    'SAGEVariant1Baseline',
    'SAGEVariant2BatchNorm',
    'SAGEVariant3Depth',
    'SAGEVariant4Both',
]

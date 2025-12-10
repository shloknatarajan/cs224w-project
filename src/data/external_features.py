"""
Unified external feature loader for OGBL-DDI.

Provides a single interface to load and combine multiple feature types
with configurable on/off controls for each feature source.

Feature Types:
- Morgan fingerprints (Phase 1): 2048-dim binary molecular substructure
- PubChem properties (Phase 2): ~9-dim continuous molecular properties
- ChemBERTa embeddings (Phase 3): 768-dim pre-trained molecular embeddings
- Drug-target features (Phase 4): Variable-dim binary target interactions
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class FeatureConfig:
    """Configuration for which features to load and combine."""

    # Phase 1: Morgan fingerprints
    use_morgan: bool = False
    morgan_path: Path = field(default_factory=lambda: DATA_DIR / "morgan_features.pt")
    morgan_dim: int = 2048

    # Phase 2: PubChem properties
    use_pubchem: bool = False
    pubchem_path: Path = field(default_factory=lambda: DATA_DIR / "ogbl_ddi_properties.csv")

    # Phase 3: ChemBERTa embeddings
    use_chemberta: bool = False
    chemberta_path: Path = field(default_factory=lambda: DATA_DIR / "chemberta_embeddings.pt")
    chemberta_dim: int = 768

    # Phase 4: Drug-target features
    use_drug_targets: bool = False
    drug_targets_path: Path = field(default_factory=lambda: DATA_DIR / "drug_target_features.pt")

    # Feature processing
    normalize: bool = True  # Normalize continuous features
    fill_missing: str = "zero"  # How to handle missing: "zero", "mean"

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.morgan_path, str):
            self.morgan_path = Path(self.morgan_path)
        if isinstance(self.pubchem_path, str):
            self.pubchem_path = Path(self.pubchem_path)
        if isinstance(self.chemberta_path, str):
            self.chemberta_path = Path(self.chemberta_path)
        if isinstance(self.drug_targets_path, str):
            self.drug_targets_path = Path(self.drug_targets_path)


@dataclass
class ExternalFeatures:
    """Container for loaded external features."""

    # Individual feature tensors (None if not loaded)
    morgan: Optional[torch.Tensor] = None
    pubchem: Optional[torch.Tensor] = None
    chemberta: Optional[torch.Tensor] = None
    drug_targets: Optional[torch.Tensor] = None

    # Combined features
    combined: Optional[torch.Tensor] = None

    # Metadata
    num_nodes: int = 0
    feature_dims: dict = field(default_factory=dict)
    total_dim: int = 0

    def to(self, device: str) -> "ExternalFeatures":
        """Move all features to device."""
        if self.morgan is not None:
            self.morgan = self.morgan.to(device)
        if self.pubchem is not None:
            self.pubchem = self.pubchem.to(device)
        if self.chemberta is not None:
            self.chemberta = self.chemberta.to(device)
        if self.drug_targets is not None:
            self.drug_targets = self.drug_targets.to(device)
        if self.combined is not None:
            self.combined = self.combined.to(device)
        return self


def load_morgan_features(
    path: Path,
    num_nodes: int,
    smiles_csv_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Load Morgan fingerprint features.

    If cached .pt file exists, load it. Otherwise, build from SMILES CSV.
    """
    if path.exists():
        logger.info(f"Loading cached Morgan features from {path}")
        features = torch.load(path, weights_only=True)
        if features.shape[0] != num_nodes:
            raise ValueError(f"Morgan features have {features.shape[0]} nodes, expected {num_nodes}")
        return features

    # Build from SMILES if available
    if smiles_csv_path and smiles_csv_path.exists():
        from .data_loader import build_smiles_feature_matrix
        logger.info(f"Building Morgan features from {smiles_csv_path}")
        features = build_smiles_feature_matrix(num_nodes, str(smiles_csv_path))
        torch.save(features, path)
        return features

    raise FileNotFoundError(
        f"Morgan features not found at {path}. "
        "Run: python -m src.data.fetch_smiles first, then re-run with smiles_csv_path."
    )


def load_pubchem_features(
    path: Path,
    num_nodes: int,
    normalize: bool = True,
    fill_missing: str = "zero",
) -> torch.Tensor:
    """Load PubChem molecular property features."""
    if not path.exists():
        raise FileNotFoundError(
            f"PubChem properties not found at {path}. "
            "Run: python -m src.data.fetch_pubchem_properties"
        )

    df = pd.read_csv(path)
    df = df.sort_values('ogb_id')

    # Get property columns (exclude ogb_id)
    prop_cols = [c for c in df.columns if c != 'ogb_id']

    # Build feature matrix
    features = np.zeros((num_nodes, len(prop_cols)), dtype=np.float32)

    for i, row in df.iterrows():
        ogb_id = int(row['ogb_id'])
        if ogb_id < num_nodes:
            for j, col in enumerate(prop_cols):
                val = row[col]
                if pd.notna(val):
                    features[ogb_id, j] = float(val)

    # Handle missing values
    if fill_missing == "mean":
        col_means = np.nanmean(features, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        for j in range(features.shape[1]):
            mask = features[:, j] == 0
            features[mask, j] = col_means[j]

    # Normalize
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    logger.info(f"Loaded PubChem features: shape={features.shape}")
    return torch.tensor(features, dtype=torch.float32)


def load_chemberta_features(path: Path, num_nodes: int) -> torch.Tensor:
    """Load ChemBERTa embedding features."""
    if not path.exists():
        raise FileNotFoundError(
            f"ChemBERTa embeddings not found at {path}. "
            "Run: python -m src.data.extract_chemberta_embeddings"
        )

    features = torch.load(path, weights_only=True)
    if features.shape[0] != num_nodes:
        raise ValueError(f"ChemBERTa features have {features.shape[0]} nodes, expected {num_nodes}")

    logger.info(f"Loaded ChemBERTa features: shape={features.shape}")
    return features


def load_drug_target_features(path: Path, num_nodes: int) -> torch.Tensor:
    """Load drug-target interaction features."""
    if not path.exists():
        raise FileNotFoundError(
            f"Drug-target features not found at {path}. "
            "Run: python -m src.data.fetch_drug_targets"
        )

    features = torch.load(path, weights_only=True)
    if features.shape[0] != num_nodes:
        raise ValueError(f"Drug-target features have {features.shape[0]} nodes, expected {num_nodes}")

    logger.info(f"Loaded drug-target features: shape={features.shape}")
    return features


def load_external_features(
    config: FeatureConfig,
    num_nodes: int,
    smiles_csv_path: Optional[Path] = None,
) -> ExternalFeatures:
    """
    Load and combine external features based on configuration.

    Args:
        config: FeatureConfig specifying which features to load
        num_nodes: Number of nodes in the graph
        smiles_csv_path: Optional path to SMILES CSV for building Morgan features

    Returns:
        ExternalFeatures container with loaded features
    """
    result = ExternalFeatures(num_nodes=num_nodes)
    feature_list = []

    # Phase 1: Morgan fingerprints
    if config.use_morgan:
        try:
            result.morgan = load_morgan_features(
                config.morgan_path, num_nodes, smiles_csv_path
            )
            result.feature_dims['morgan'] = result.morgan.shape[1]
            feature_list.append(result.morgan)
            logger.info(f"  Morgan fingerprints: {result.morgan.shape[1]} dims")
        except FileNotFoundError as e:
            logger.warning(f"  Morgan fingerprints: SKIPPED ({e})")

    # Phase 2: PubChem properties
    if config.use_pubchem:
        try:
            result.pubchem = load_pubchem_features(
                config.pubchem_path, num_nodes,
                normalize=config.normalize,
                fill_missing=config.fill_missing
            )
            result.feature_dims['pubchem'] = result.pubchem.shape[1]
            feature_list.append(result.pubchem)
            logger.info(f"  PubChem properties: {result.pubchem.shape[1]} dims")
        except FileNotFoundError as e:
            logger.warning(f"  PubChem properties: SKIPPED ({e})")

    # Phase 3: ChemBERTa embeddings
    if config.use_chemberta:
        try:
            result.chemberta = load_chemberta_features(
                config.chemberta_path, num_nodes
            )
            result.feature_dims['chemberta'] = result.chemberta.shape[1]
            feature_list.append(result.chemberta)
            logger.info(f"  ChemBERTa embeddings: {result.chemberta.shape[1]} dims")
        except FileNotFoundError as e:
            logger.warning(f"  ChemBERTa embeddings: SKIPPED ({e})")

    # Phase 4: Drug-target features
    if config.use_drug_targets:
        try:
            result.drug_targets = load_drug_target_features(
                config.drug_targets_path, num_nodes
            )
            result.feature_dims['drug_targets'] = result.drug_targets.shape[1]
            feature_list.append(result.drug_targets)
            logger.info(f"  Drug-target features: {result.drug_targets.shape[1]} dims")
        except FileNotFoundError as e:
            logger.warning(f"  Drug-target features: SKIPPED ({e})")

    # Combine features
    if feature_list:
        result.combined = torch.cat(feature_list, dim=1)
        result.total_dim = result.combined.shape[1]
        logger.info(f"Combined features: {result.total_dim} dims total")
    else:
        logger.warning("No external features loaded!")
        result.total_dim = 0

    return result


def get_default_config() -> FeatureConfig:
    """Get default feature configuration (all features enabled)."""
    return FeatureConfig(
        use_morgan=True,
        use_pubchem=True,
        use_chemberta=True,
        use_drug_targets=True,
    )


def get_minimal_config() -> FeatureConfig:
    """Get minimal configuration (Morgan fingerprints only)."""
    return FeatureConfig(
        use_morgan=True,
        use_pubchem=False,
        use_chemberta=False,
        use_drug_targets=False,
    )


# Convenience function for quick testing
if __name__ == "__main__":
    # Test loading with all features
    config = get_default_config()
    print(f"Testing feature loading with config:")
    print(f"  use_morgan: {config.use_morgan}")
    print(f"  use_pubchem: {config.use_pubchem}")
    print(f"  use_chemberta: {config.use_chemberta}")
    print(f"  use_drug_targets: {config.use_drug_targets}")

    try:
        features = load_external_features(config, num_nodes=4267)
        print(f"\nLoaded features:")
        print(f"  Total dimension: {features.total_dim}")
        print(f"  Feature dims: {features.feature_dims}")
    except Exception as e:
        print(f"\nError loading features: {e}")
        print("Some feature files may not exist yet. Run the fetch scripts first.")

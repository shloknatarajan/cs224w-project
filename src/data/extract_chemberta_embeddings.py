"""
Extract ChemBERTa embeddings for OGBL-DDI drugs.

ChemBERTa is a transformer model pre-trained on ~77M SMILES strings.
It captures rich chemical semantics that fingerprints miss.

Model: seyonec/ChemBERTa-zinc-base-v1 (768-dim embeddings)
"""
import logging
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SMILES_PATH = Path(__file__).parent.parent.parent / "data/ogbl_ddi_smiles.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent / "data/chemberta_embeddings.pt"

# ChemBERTa model
CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
EMBEDDING_DIM = 768


def extract_chemberta_embeddings(
    smiles_path: Path = DEFAULT_SMILES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = CHEMBERTA_MODEL,
    batch_size: int = 32,
    device: str = None,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Extract ChemBERTa embeddings for all drugs with SMILES.

    Args:
        smiles_path: Path to SMILES CSV (from fetch_smiles.py)
        output_path: Where to save the embeddings tensor
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        device: 'cuda', 'mps', or 'cpu' (auto-detect if None)
        max_length: Maximum SMILES sequence length

    Returns:
        Tensor of shape [num_nodes, 768] with molecular embeddings
    """
    # Import here to avoid loading transformers unless needed
    from transformers import AutoTokenizer, AutoModel

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Load SMILES data
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"SMILES file not found at {smiles_path}. "
            "Run fetch_smiles.py first."
        )

    smiles_df = pd.read_csv(smiles_path)
    num_nodes = len(smiles_df)
    has_smiles = smiles_df['smiles'].notna().sum()
    logger.info(f"Loaded {num_nodes} drugs, {has_smiles} have SMILES")

    # Load model and tokenizer
    logger.info(f"Loading ChemBERTa model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Initialize embeddings tensor
    embeddings = torch.zeros(num_nodes, EMBEDDING_DIM)

    # Get drugs with valid SMILES
    valid_mask = smiles_df['smiles'].notna()
    valid_df = smiles_df[valid_mask].reset_index(drop=True)

    logger.info(f"Extracting embeddings for {len(valid_df)} drugs with SMILES...")

    with torch.no_grad():
        for start in tqdm(range(0, len(valid_df), batch_size), desc="Extracting"):
            end = min(start + batch_size, len(valid_df))
            batch_df = valid_df.iloc[start:end]

            smiles_list = batch_df['smiles'].tolist()
            node_ids = batch_df['ogb_id'].tolist()

            # Tokenize SMILES
            inputs = tokenizer(
                smiles_list,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            # Get embeddings
            outputs = model(**inputs)

            # Mean pooling over sequence (excluding padding)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            # Expand mask for broadcasting
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            # Mean pooling
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

            # Store in output tensor
            for i, node_id in enumerate(node_ids):
                embeddings[node_id] = pooled[i].cpu()

    # Save embeddings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"Shape: {embeddings.shape}, Non-zero rows: {(embeddings.sum(dim=1) != 0).sum()}")

    return embeddings


def load_chemberta_embeddings(path: Path = DEFAULT_OUTPUT_PATH) -> torch.Tensor:
    """Load pre-computed ChemBERTa embeddings."""
    if not path.exists():
        raise FileNotFoundError(
            f"ChemBERTa embeddings not found at {path}. "
            "Run extract_chemberta_embeddings() first."
        )
    return torch.load(path, weights_only=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract ChemBERTa embeddings for OGBL-DDI drugs")
    parser.add_argument("--smiles", type=Path, default=DEFAULT_SMILES_PATH,
                        help="Path to SMILES CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="Output tensor path (.pt)")
    parser.add_argument("--model", type=str, default=CHEMBERTA_MODEL,
                        help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/mps/cpu)")

    args = parser.parse_args()

    extract_chemberta_embeddings(
        smiles_path=args.smiles,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

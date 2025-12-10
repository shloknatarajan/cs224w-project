import os
import torch
import networkx as nx
import logging
import pandas as pd
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import degree, to_networkx, add_self_loops
from torch_geometric.data import Data as PyGData
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

def smiles_to_morgan(smiles: str, n_bits: int = 2048, radius: int = 2) -> torch.Tensor:
    """
    Convert a SMILES string to a Morgan/ECFP fingerprint vector [n_bits].
    Returns an all-zero vector if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(n_bits, dtype=torch.float32)

    bitvect = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
    arr = torch.zeros(n_bits, dtype=torch.float32)
    on_bits = list(bitvect.GetOnBits())
    arr[on_bits] = 1.0
    return arr


def build_smiles_feature_matrix(
    num_nodes: int,
    smiles_csv_path: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> torch.Tensor:
    """
    Build a [num_nodes, n_bits] feature matrix from a CSV with columns:
      - ogb_id  (int: 0..num_nodes-1)
      - smiles  (str)
    Nodes without a SMILES entry (or invalid SMILES) get all-zero features.
    """
    df = pd.read_csv(smiles_csv_path, keep_default_na=False)
    if "ogb_id" not in df.columns or "smiles" not in df.columns:
        raise ValueError(
            f"Expected columns 'ogb_id' and 'smiles' in {smiles_csv_path}, "
            f"found {df.columns.tolist()}"
        )

    smiles_by_id = dict(
        zip(df["ogb_id"].astype(int), df["smiles"].astype(str))
    )

    feat = torch.zeros((num_nodes, n_bits), dtype=torch.float32)
    missing = 0

    for node_id in range(num_nodes):
        smi = smiles_by_id.get(node_id, "")
        # Skip empty strings, "nan", or invalid SMILES
        if not smi or smi == "nan" or not isinstance(smi, str) or len(smi) == 0:
            missing += 1
            continue
        feat[node_id] = smiles_to_morgan(smi, n_bits=n_bits, radius=radius)

    logger.info(
        f"Built SMILES feature matrix: shape={feat.shape}, "
        f"missing/invalid SMILES for {missing}/{num_nodes} nodes"
    )
    return feat


def smiles_to_chemberta(
    smiles: str,
    tokenizer,
    model,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Convert a SMILES string to a ChemBERTa embedding vector.
    Returns an all-zero vector if SMILES is invalid.

    Args:
        smiles: SMILES string
        tokenizer: ChemBERTa tokenizer
        model: ChemBERTa model
        device: Device to run on

    Returns:
        Embedding vector of shape [hidden_dim] (usually 768)
    """
    try:
        # Tokenize the SMILES string
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings from ChemBERTa
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

        return embedding.cpu()
    except Exception as e:
        logger.warning(f"Failed to encode SMILES '{smiles}': {e}")
        # Return zero vector with correct dimension (768 for ChemBERTa)
        return torch.zeros(768, dtype=torch.float32)


def build_chemberta_feature_matrix(
    num_nodes: int,
    smiles_csv_path: str,
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    device: str = "cpu",
    batch_size: int = 32,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Build a [num_nodes, embedding_dim] feature matrix using ChemBERTa embeddings.

    Args:
        num_nodes: Number of nodes in the graph
        smiles_csv_path: Path to CSV with columns ['ogb_id', 'smiles']
        model_name: ChemBERTa model name from HuggingFace
        device: Device to run on
        batch_size: Batch size for processing SMILES
        return_mask: If True, also return a mask indicating which nodes have valid SMILES

    Returns:
        If return_mask=False: Feature matrix of shape [num_nodes, embedding_dim]
        If return_mask=True: Tuple of (feature matrix, mask) where mask is [num_nodes] with 1=valid, 0=missing
    """
    logger.info(f"Loading ChemBERTa model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    logger.info(f"ChemBERTa embedding dimension: {embedding_dim}")

    # Load SMILES data
    df = pd.read_csv(smiles_csv_path, keep_default_na=False)
    if "ogb_id" not in df.columns or "smiles" not in df.columns:
        raise ValueError(
            f"Expected columns 'ogb_id' and 'smiles' in {smiles_csv_path}, "
            f"found {df.columns.tolist()}"
        )

    smiles_by_id = dict(
        zip(df["ogb_id"].astype(int), df["smiles"].astype(str))
    )

    # Initialize feature matrix and mask
    feat = torch.zeros((num_nodes, embedding_dim), dtype=torch.float32)
    mask = torch.zeros(num_nodes, dtype=torch.float32)  # 0 = missing, 1 = valid
    missing = 0

    # Process in batches for efficiency
    logger.info(f"Processing {num_nodes} nodes with batch_size={batch_size}...")
    for start_idx in range(0, num_nodes, batch_size):
        end_idx = min(start_idx + batch_size, num_nodes)
        batch_smiles = []
        batch_indices = []

        for node_id in range(start_idx, end_idx):
            smi = smiles_by_id.get(node_id, "")
            if not smi or smi == "nan" or not isinstance(smi, str) or len(smi) == 0:
                missing += 1
                continue
            batch_smiles.append(smi)
            batch_indices.append(node_id)

        if batch_smiles:
            # Batch tokenize
            inputs = tokenizer(
                batch_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :].cpu()

            # Store embeddings and mark as valid
            for idx, node_id in enumerate(batch_indices):
                feat[node_id] = embeddings[idx]
                mask[node_id] = 1.0

        if (start_idx // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {end_idx}/{num_nodes} nodes...")

    logger.info(
        f"Built ChemBERTa feature matrix: shape={feat.shape}, "
        f"missing/invalid SMILES for {missing}/{num_nodes} nodes"
    )

    if return_mask:
        logger.info(f"Returning mask: {mask.sum().item():.0f} valid, {(num_nodes - mask.sum().item()):.0f} missing")
        return feat, mask
    return feat


def load_dataset(
    dataset_name: str = "ogbl-ddi",
    device: str = "cpu",
    smiles_csv_path: str | None = None,
    feature_cache_path: str | None = None,
    fp_n_bits: int = 2048,
    fp_radius: int = 2,
):
    """
    Load OGB link prediction dataset and split edges, optionally attaching
    SMILES-based Morgan fingerprint features as data.x.

    Args:
        dataset_name: OGB dataset name (default: 'ogbl-ddi').
        device: 'cpu' or 'cuda'.
        smiles_csv_path: path to CSV with columns ['ogb_id', 'smiles'].
        feature_cache_path: optional .pt path to cache/load feature matrix.
        fp_n_bits: length of Morgan fingerprint.
        fp_radius: radius for Morgan fingerprint.

    Returns:
        tuple: (data, split_edge, num_nodes, evaluator)
    """
    logger.info(f"Loading dataset {dataset_name}...")
    dataset = PygLinkPropPredDataset(dataset_name)
    data = dataset[0]

    split_edge = dataset.get_edge_split()
    logger.info(f"Dataset loaded: {data.num_nodes} nodes")

    # Move edge splits to device (you already had this)
    train_pos = split_edge["train"]["edge"].to(device)
    valid_pos = split_edge["valid"]["edge"].to(device)
    valid_neg = split_edge["valid"]["edge_neg"].to(device)
    test_pos = split_edge["test"]["edge"].to(device)
    test_neg = split_edge["test"]["edge_neg"].to(device)

    logger.info(
        f"Train pos edges: {train_pos.size(0)}, "
        f"Valid pos: {valid_pos.size(0)}, Test pos: {test_pos.size(0)}"
    )
    logger.info(
        f"Valid neg edges: {valid_neg.size(0)}, "
        f"Test neg: {test_neg.size(0)}"
    )

    num_nodes = data.num_nodes

    # Construct graph using only training edges
    train_edge_index = train_pos.t().contiguous().to(device)

    # Add self-loops
    data.edge_index, _ = add_self_loops(train_edge_index, num_nodes=num_nodes)
    logger.info(
        f"Added self-loops: Total edges now = {data.edge_index.size(1)}"
    )

    if smiles_csv_path is not None:
        if feature_cache_path is not None and os.path.exists(feature_cache_path):
            logger.info(f"Loading cached SMILES features from {feature_cache_path}")
            feat = torch.load(feature_cache_path, map_location=device)
        else:
            logger.info(f"Building SMILES features from {smiles_csv_path}...")
            feat = build_smiles_feature_matrix(
                num_nodes=num_nodes,
                smiles_csv_path=smiles_csv_path,
                n_bits=fp_n_bits,
                radius=fp_radius,
            )
            if feature_cache_path is not None:
                torch.save(feat.cpu(), feature_cache_path)
                logger.info(f"Saved SMILES features to {feature_cache_path}")

        if feat.size(0) != num_nodes:
            raise ValueError(
                f"Feature matrix first dim ({feat.size(0)}) "
                f"does not match num_nodes ({num_nodes})."
            )
        data.x = feat.to(device)
        logger.info(f"Attached SMILES features: data.x shape = {data.x.shape}")
    else:
        logger.info("No smiles_csv_path provided → data.x will not be set.")

    evaluator = Evaluator(name=dataset_name)
    return data, split_edge, num_nodes, evaluator


def compute_structural_features(train_edge_index, num_nodes, device='cpu'):
    """
    Compute rich structural features for nodes.

    Features computed:
    1. Node degree (log-normalized)
    2. Clustering coefficient
    3. Core number (k-core decomposition)
    4. PageRank
    5. Neighbor degree statistics (mean and max)

    Args:
        train_edge_index: Training edge index (2 x E)
        num_nodes: Number of nodes
        device: Device to put features on

    Returns:
        torch.Tensor: Structural features of shape (num_nodes, 6)
    """
    logger.info("Computing structural features (this may take a moment)...")

    # Convert to NetworkX graph for feature computation
    edge_index_cpu = train_edge_index.cpu()
    G = to_networkx(PyGData(edge_index=edge_index_cpu, num_nodes=num_nodes), to_undirected=True)

    # 1. Node degree (log-normalized)
    node_degrees = degree(train_edge_index[0], num_nodes=num_nodes, dtype=torch.float).to(device)
    degree_features = torch.log(node_degrees + 1).unsqueeze(1)

    # 2. Clustering coefficient
    clustering = nx.clustering(G)
    clustering_features = torch.tensor([clustering[i] for i in range(num_nodes)], dtype=torch.float).unsqueeze(1).to(device)

    # 3. Core number
    core_number = nx.core_number(G)
    core_features = torch.tensor([core_number[i] for i in range(num_nodes)], dtype=torch.float).unsqueeze(1).to(device)
    core_features = torch.log(core_features + 1)

    # 4. PageRank
    pagerank = nx.pagerank(G, max_iter=50)
    pagerank_features = torch.tensor([pagerank[i] for i in range(num_nodes)], dtype=torch.float).unsqueeze(1).to(device)
    pagerank_features = pagerank_features / pagerank_features.max()

    # 5. Neighbor degree statistics
    row, col = train_edge_index
    neighbor_degrees = torch.zeros((num_nodes, 2), dtype=torch.float).to(device)
    for node in range(num_nodes):
        neighbors = col[row == node]
        if len(neighbors) > 0:
            neighbor_degs = node_degrees[neighbors]
            neighbor_degrees[node, 0] = neighbor_degs.mean()
            neighbor_degrees[node, 1] = neighbor_degs.max()
    neighbor_degrees = torch.log(neighbor_degrees + 1)

    # Combine all features
    node_structural_features = torch.cat([
        degree_features,
        clustering_features,
        core_features,
        pagerank_features,
        neighbor_degrees
    ], dim=1)

    logger.info(f"Computed structural features ({node_structural_features.shape[1]} dims):")
    logger.info(f"  - Degree: mean={degree_features.mean():.3f}, std={degree_features.std():.3f}")
    logger.info(f"  - Clustering: mean={clustering_features.mean():.3f}, std={clustering_features.std():.3f}")
    logger.info(f"  - Core number: mean={core_features.mean():.3f}, std={core_features.std():.3f}")
    logger.info(f"  - PageRank: mean={pagerank_features.mean():.3f}, std={pagerank_features.std():.3f}")

    return node_structural_features


def load_dataset_chemberta(
    dataset_name: str = "ogbl-ddi",
    device: str = "cpu",
    smiles_csv_path: str | None = None,
    feature_cache_path: str | None = None,
    chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 32,
):
    """
    Load OGB link prediction dataset and split edges, attaching ChemBERTa embeddings as data.x.

    Args:
        dataset_name: OGB dataset name (default: 'ogbl-ddi').
        device: 'cpu' or 'cuda'.
        smiles_csv_path: path to CSV with columns ['ogb_id', 'smiles'].
        feature_cache_path: optional .pt path to cache/load feature matrix.
        chemberta_model: HuggingFace model name for ChemBERTa.
        batch_size: Batch size for processing SMILES strings.

    Returns:
        tuple: (data, split_edge, num_nodes, evaluator)
    """
    logger.info(f"Loading dataset {dataset_name}...")
    dataset = PygLinkPropPredDataset(dataset_name)
    data = dataset[0]

    split_edge = dataset.get_edge_split()
    logger.info(f"Dataset loaded: {data.num_nodes} nodes")

    # Move edge splits to device
    train_pos = split_edge["train"]["edge"].to(device)
    valid_pos = split_edge["valid"]["edge"].to(device)
    valid_neg = split_edge["valid"]["edge_neg"].to(device)
    test_pos = split_edge["test"]["edge"].to(device)
    test_neg = split_edge["test"]["edge_neg"].to(device)

    logger.info(
        f"Train pos edges: {train_pos.size(0)}, "
        f"Valid pos: {valid_pos.size(0)}, Test pos: {test_pos.size(0)}"
    )
    logger.info(
        f"Valid neg edges: {valid_neg.size(0)}, "
        f"Test neg: {test_neg.size(0)}"
    )

    num_nodes = data.num_nodes

    # Construct graph using only training edges
    train_edge_index = train_pos.t().contiguous().to(device)

    # Compute structural features for fallback
    structural_features = compute_structural_features(train_edge_index, num_nodes, device=device)

    # Add self-loops
    data.edge_index, _ = add_self_loops(train_edge_index, num_nodes=num_nodes)
    logger.info(
        f"Added self-loops: Total edges now = {data.edge_index.size(1)}"
    )

    if smiles_csv_path is not None:
        # Check for cached features and mask
        mask_cache_path = feature_cache_path.replace('.pt', '_mask.pt') if feature_cache_path else None

        if feature_cache_path is not None and os.path.exists(feature_cache_path):
            logger.info(f"Loading cached ChemBERTa features from {feature_cache_path}")
            feat = torch.load(feature_cache_path, map_location=device, weights_only=True)
            if mask_cache_path and os.path.exists(mask_cache_path):
                logger.info(f"Loading cached SMILES mask from {mask_cache_path}")
                smiles_mask = torch.load(mask_cache_path, map_location=device, weights_only=True)
            else:
                logger.warning("Cached features found but no mask - computing mask from SMILES CSV")
                # Compute mask from SMILES CSV
                import pandas as pd
                df = pd.read_csv(smiles_csv_path, keep_default_na=False)
                smiles_mask = torch.zeros(num_nodes, dtype=torch.float32)
                for _, row in df.iterrows():
                    ogb_id = int(row['ogb_id'])
                    smiles = str(row['smiles'])
                    if smiles and len(smiles) > 0 and smiles != 'nan':
                        smiles_mask[ogb_id] = 1.0
                if mask_cache_path:
                    torch.save(smiles_mask.cpu(), mask_cache_path)
                    logger.info(f"Saved SMILES mask to {mask_cache_path}")
        else:
            logger.info(f"Building ChemBERTa features from {smiles_csv_path}...")
            feat, smiles_mask = build_chemberta_feature_matrix(
                num_nodes=num_nodes,
                smiles_csv_path=smiles_csv_path,
                model_name=chemberta_model,
                device=device,
                batch_size=batch_size,
                return_mask=True,
            )
            if feature_cache_path is not None:
                torch.save(feat.cpu(), feature_cache_path)
                logger.info(f"Saved ChemBERTa features to {feature_cache_path}")
            if mask_cache_path is not None:
                torch.save(smiles_mask.cpu(), mask_cache_path)
                logger.info(f"Saved SMILES mask to {mask_cache_path}")

        if feat.size(0) != num_nodes:
            raise ValueError(
                f"Feature matrix first dim ({feat.size(0)}) "
                f"does not match num_nodes ({num_nodes})."
            )
        data.x = feat.to(device)
        data.smiles_mask = smiles_mask.to(device)
        logger.info(f"Attached ChemBERTa features: data.x shape = {data.x.shape}")
        logger.info(f"Attached SMILES mask: {smiles_mask.sum().item():.0f} valid / {num_nodes} total ({smiles_mask.sum().item()/num_nodes*100:.1f}%)")
    else:
        logger.info("No smiles_csv_path provided → data.x will not be set.")

    # Attach structural features for nodes without SMILES (and as auxiliary signal)
    data.struct_features = structural_features
    logger.info(f"Attached structural features: data.struct_features shape = {structural_features.shape}")

    evaluator = Evaluator(name=dataset_name)
    return data, split_edge, num_nodes, evaluator

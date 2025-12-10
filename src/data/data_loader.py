import torch
import networkx as nx
import logging
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import degree, to_networkx, add_self_loops
from torch_geometric.data import Data as PyGData
from rdkit import Chem
from rdkit.Chem import AllChem

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
    df = pd.read_csv(smiles_csv_path)
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
        if not smi or not isinstance(smi, str):
            missing += 1
            continue
        feat[node_id] = smiles_to_morgan(smi, n_bits=n_bits, radius=radius)

    logger.info(
        f"Built SMILES feature matrix: shape={feat.shape}, "
        f"missing/invalid SMILES for {missing}/{num_nodes} nodes"
    )
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


def load_dataset(dataset_name='ogbl-ddi', device='cpu'):
    """
    Load OGB link prediction dataset and split edges.

    Returns:
        tuple: (data, split_edge, num_nodes, evaluator)
    """
    logger.info(f"Loading dataset {dataset_name}...")
    dataset = PygLinkPropPredDataset(dataset_name)
    data = dataset[0]

    split_edge = dataset.get_edge_split()
    logger.info(f"Dataset loaded: {data.num_nodes} nodes")

    # Move data to device
    train_pos = split_edge['train']['edge'].to(device)
    valid_pos = split_edge['valid']['edge'].to(device)
    valid_neg = split_edge['valid']['edge_neg'].to(device)
    test_pos = split_edge['test']['edge'].to(device)
    test_neg = split_edge['test']['edge_neg'].to(device)

    logger.info(f"Train pos edges: {train_pos.size(0)}, Valid pos: {valid_pos.size(0)}, Test pos: {test_pos.size(0)}")
    logger.info(f"Valid neg edges: {valid_neg.size(0)}, Test neg: {test_neg.size(0)}")

    num_nodes = data.num_nodes

    # Construct graph using only training edges
    train_edge_index = train_pos.t().contiguous().to(device)

    # Add self-loops
    data.edge_index, _ = add_self_loops(train_edge_index, num_nodes=num_nodes)
    logger.info(f"Added self-loops: Total edges now = {data.edge_index.size(1)}")

    from ogb.linkproppred import Evaluator
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

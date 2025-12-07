import torch
import networkx as nx
import logging
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import degree, to_networkx, add_self_loops
from torch_geometric.data import Data as PyGData

logger = logging.getLogger(__name__)


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

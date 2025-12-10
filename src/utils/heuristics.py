"""
Graph heuristics for link prediction (simplified leaderboard approach)

Implements common link prediction heuristics:
- Common Neighbors (CN)
- Jaccard Coefficient
- Adamic-Adar (AA)
- Resource Allocation (RA)
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def build_adjacency_list(edge_index, num_nodes):
    """
    Build adjacency list from edge index.

    Args:
        edge_index: [2, E] edge index tensor
        num_nodes: Number of nodes

    Returns:
        adj_list: Dict mapping node -> list of neighbors
    """
    adj_list = defaultdict(set)

    edges = edge_index.t().cpu().numpy()  # [E, 2]
    for src, dst in edges:
        adj_list[src].add(dst)
        adj_list[dst].add(src)  # Undirected

    return adj_list


def compute_edge_heuristics(edge_index, edges_to_score, num_nodes, adj_list=None):
    """
    Compute graph heuristics for given edges.

    Args:
        edge_index: Training graph edges [2, E]
        edges_to_score: Edges to compute heuristics for [N, 2]
        num_nodes: Number of nodes in graph
        adj_list: Pre-computed adjacency list (optional, will build if not provided)

    Returns:
        heuristics: Tensor [N, 4] with [CN, Jaccard, AA, RA] for each edge
    """
    # Build adjacency list (or use pre-computed)
    if adj_list is None:
        adj_list = build_adjacency_list(edge_index, num_nodes)

    # Compute degrees
    degrees = np.zeros(num_nodes, dtype=np.float32)
    for node, neighbors in adj_list.items():
        degrees[node] = len(neighbors)

    src = edges_to_score[:, 0].cpu().numpy()
    dst = edges_to_score[:, 1].cpu().numpy()

    cn_scores = []
    jaccard_scores = []
    aa_scores = []
    ra_scores = []

    for i in range(len(src)):
        s, d = src[i], dst[i]

        # Get neighbors
        neighbors_s = adj_list[s]
        neighbors_d = adj_list[d]

        # Common neighbors (CN)
        common = neighbors_s & neighbors_d
        cn = len(common)
        cn_scores.append(cn)

        # Jaccard coefficient
        union = neighbors_s | neighbors_d
        jaccard = cn / len(union) if len(union) > 0 else 0.0
        jaccard_scores.append(jaccard)

        # Adamic-Adar (AA) and Resource Allocation (RA)
        aa = 0.0
        ra = 0.0
        for neighbor in common:
            deg = degrees[neighbor]
            if deg > 1:
                aa += 1.0 / np.log(deg)
            if deg > 0:
                ra += 1.0 / deg
        aa_scores.append(aa)
        ra_scores.append(ra)

    # Stack into tensor [N, 4]
    heuristics = torch.tensor([
        cn_scores,
        jaccard_scores,
        aa_scores,
        ra_scores
    ], dtype=torch.float32).t()  # Transpose to [N, 4]

    return heuristics


def compute_edge_heuristics_batched(edge_index, edges_to_score, num_nodes, batch_size=10000, device='cpu'):
    """
    Compute heuristics in batches to handle large edge sets.

    Args:
        edge_index: Training graph edges [2, E]
        edges_to_score: Edges to compute heuristics for [N, 2]
        num_nodes: Number of nodes
        batch_size: Batch size for processing
        device: Device to return results on

    Returns:
        heuristics: Tensor [N, 4] on specified device
    """
    # Build adjacency list ONCE (huge speedup!)
    adj_list = build_adjacency_list(edge_index, num_nodes)
    
    all_heuristics = []

    for start in range(0, edges_to_score.size(0), batch_size):
        end = min(start + batch_size, edges_to_score.size(0))
        batch_edges = edges_to_score[start:end]

        batch_heuristics = compute_edge_heuristics(edge_index, batch_edges, num_nodes, adj_list=adj_list)
        all_heuristics.append(batch_heuristics)

    heuristics = torch.cat(all_heuristics, dim=0)
    return heuristics.to(device)


def normalize_heuristics(heuristics, method='log1p'):
    """
    Normalize heuristics to reasonable ranges.

    Args:
        heuristics: [N, 4] tensor
        method: 'log1p', 'standardize', or 'minmax'

    Returns:
        normalized: [N, 4] tensor
    """
    if method == 'log1p':
        # Log transform (common for count-based features like CN)
        return torch.log1p(heuristics)
    elif method == 'standardize':
        # Standardize to mean=0, std=1
        mean = heuristics.mean(dim=0, keepdim=True)
        std = heuristics.std(dim=0, keepdim=True) + 1e-8
        return (heuristics - mean) / std
    elif method == 'minmax':
        # Scale to [0, 1]
        min_val = heuristics.min(dim=0, keepdim=True)[0]
        max_val = heuristics.max(dim=0, keepdim=True)[0]
        return (heuristics - min_val) / (max_val - min_val + 1e-8)
    else:
        return heuristics


def discretize_heuristics(heuristics, max_bins=100):
    """
    Discretize continuous heuristics into bins for embedding lookup.
    
    This matches the leaderboard approach which uses embedding tables
    instead of continuous values.
    
    Args:
        heuristics: [N, 4] tensor with continuous values [CN, Jaccard, AA, RA]
        max_bins: Maximum number of bins (default 100, matches leaderboard)
    
    Returns:
        discretized: [N, 4] LongTensor with bin indices [0, max_bins-1]
    """
    discretized = torch.zeros_like(heuristics, dtype=torch.long)
    
    # CN: Count-based, use direct binning (typically 0-20 range for DDI)
    cn = heuristics[:, 0]
    discretized[:, 0] = torch.clamp(cn.long(), 0, max_bins - 1)
    
    # Jaccard: [0, 1] range, bin into [0, max_bins-1]
    jaccard = heuristics[:, 1]
    discretized[:, 1] = torch.clamp((jaccard * max_bins).long(), 0, max_bins - 1)
    
    # AA: Continuous, log-scale then bin
    aa = heuristics[:, 2]
    # Log1p to handle 0 values, then scale to bins
    aa_logged = torch.log1p(aa)
    aa_max = aa_logged.max() + 1e-8
    discretized[:, 2] = torch.clamp((aa_logged / aa_max * max_bins).long(), 0, max_bins - 1)
    
    # RA: Similar to AA
    ra = heuristics[:, 3]
    ra_logged = torch.log1p(ra)
    ra_max = ra_logged.max() + 1e-8
    discretized[:, 3] = torch.clamp((ra_logged / ra_max * max_bins).long(), 0, max_bins - 1)
    
    return discretized
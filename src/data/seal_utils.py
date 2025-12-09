"""
SEAL (Subgraph Extraction and Labeling) Utilities
Implements subgraph extraction and DRNL node labeling for link prediction
"""
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def drnl_node_labeling(edge_index, src, dst, num_nodes, max_z=1000):
    """
    Double Radius Node Labeling (DRNL) for a link (src, dst).

    Labels nodes based on their structural role relative to the target link:
    - Label 1: src and dst nodes
    - Label 2+: Other nodes based on shortest path distances d_src and d_dst
      - If d_src == d_dst: label = 1 + min(d_src, d_dst) + (d_src + d_dst - 1) * (d_src + d_dst) / 2
      - If d_src != d_dst: label = 1 + max(d_src, d_dst) + (d_src + d_dst - 1) * (d_src + d_dst) / 2 + abs(d_src - d_dst) - 1

    Args:
        edge_index: Graph edge index (2 x E)
        src: Source node
        dst: Destination node
        num_nodes: Total number of nodes
        max_z: Maximum label value (for limiting)

    Returns:
        torch.Tensor: Node labels (num_nodes,)
    """
    # Build adjacency list
    adj_list = defaultdict(set)
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[u].add(v)
        adj_list[v].add(u)

    # BFS from src
    src_distances = {src: 0}
    queue = [src]
    visited = {src}
    for node in queue:
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                src_distances[neighbor] = src_distances[node] + 1
                queue.append(neighbor)

    # BFS from dst
    dst_distances = {dst: 0}
    queue = [dst]
    visited = {dst}
    for node in queue:
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                dst_distances[neighbor] = dst_distances[node] + 1
                queue.append(neighbor)

    # Compute DRNL labels
    z = torch.zeros(num_nodes, dtype=torch.long)

    for node in range(num_nodes):
        if node == src or node == dst:
            z[node] = 1
        elif node in src_distances and node in dst_distances:
            d_src = src_distances[node]
            d_dst = dst_distances[node]

            if d_src == d_dst:
                # Same distance from both endpoints
                label = 1 + min(d_src, d_dst) + (d_src + d_dst - 1) * (d_src + d_dst) // 2
            else:
                # Different distances
                label = 1 + max(d_src, d_dst) + (d_src + d_dst - 1) * (d_src + d_dst) // 2 + abs(d_src - d_dst) - 1

            z[node] = min(label, max_z)
        else:
            # Node not reachable from src or dst
            z[node] = 0

    return z


def extract_enclosing_subgraph(edge_index, src, dst, num_nodes, num_hops=1, max_nodes_per_hop=None):
    """
    Extract k-hop enclosing subgraph around a link (src, dst).

    Args:
        edge_index: Graph edge index (2 x E)
        src: Source node
        dst: Destination node
        num_nodes: Total number of nodes
        num_hops: Number of hops for subgraph extraction
        max_nodes_per_hop: Maximum nodes to include per hop (for scalability)

    Returns:
        Data: PyG Data object containing the subgraph with node labels
    """
    # Remove the target link if it exists (for training on positive links)
    mask = ~((edge_index[0] == src) & (edge_index[1] == dst))
    mask = mask & ~((edge_index[0] == dst) & (edge_index[1] == src))
    edge_index_no_link = edge_index[:, mask]

    # Extract k-hop subgraph around both nodes
    nodes_src, edge_index_src, mapping_src, edge_mask_src = k_hop_subgraph(
        [src], num_hops, edge_index_no_link, relabel_nodes=False, num_nodes=num_nodes
    )

    nodes_dst, edge_index_dst, mapping_dst, edge_mask_dst = k_hop_subgraph(
        [dst], num_hops, edge_index_no_link, relabel_nodes=False, num_nodes=num_nodes
    )

    # Merge the two subgraphs
    nodes = torch.cat([nodes_src, nodes_dst]).unique()

    # Limit nodes if specified
    if max_nodes_per_hop is not None and len(nodes) > max_nodes_per_hop * num_hops * 2:
        # Keep nodes closest to src and dst
        nodes = nodes[:max_nodes_per_hop * num_hops * 2]

    # Extract subgraph with merged nodes
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[nodes] = True

    edge_mask = node_mask[edge_index_no_link[0]] & node_mask[edge_index_no_link[1]]
    sub_edge_index = edge_index_no_link[:, edge_mask]

    # Relabel nodes to be contiguous [0, num_subgraph_nodes)
    node_idx = torch.zeros(num_nodes, dtype=torch.long)
    node_idx[nodes] = torch.arange(len(nodes))
    sub_edge_index = node_idx[sub_edge_index]

    # Get new indices for src and dst
    src_new = node_idx[src].item()
    dst_new = node_idx[dst].item()

    # Compute DRNL labels
    z = drnl_node_labeling(sub_edge_index, src_new, dst_new, len(nodes))

    # Create data object
    data = Data(
        x=z,  # Node features are DRNL labels
        edge_index=sub_edge_index,
        num_nodes=len(nodes),
        src=src_new,
        dst=dst_new
    )

    return data


class SEALDataset(torch.utils.data.Dataset):
    """
    Dataset for SEAL that extracts subgraphs on-the-fly.
    """
    def __init__(self, edge_index, pos_edges, neg_edges, num_nodes, num_hops=1, max_nodes_per_hop=None):
        """
        Args:
            edge_index: Full graph edge index (training edges only)
            pos_edges: Positive edges to create subgraphs for (N x 2)
            neg_edges: Negative edges to create subgraphs for (M x 2)
            num_nodes: Total number of nodes
            num_hops: Number of hops for subgraph extraction
            max_nodes_per_hop: Maximum nodes per hop
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_hops = num_hops
        self.max_nodes_per_hop = max_nodes_per_hop

        # Combine positive and negative edges
        self.links = torch.cat([pos_edges, neg_edges], dim=0)
        self.labels = torch.cat([
            torch.ones(len(pos_edges)),
            torch.zeros(len(neg_edges))
        ], dim=0)

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        """Extract subgraph for the idx-th link"""
        link = self.links[idx]
        label = self.labels[idx]

        src, dst = link[0].item(), link[1].item()

        # Extract enclosing subgraph
        data = extract_enclosing_subgraph(
            self.edge_index, src, dst, self.num_nodes,
            self.num_hops, self.max_nodes_per_hop
        )

        data.y = label

        return data


def seal_collate_fn(batch):
    """
    Custom collate function for batching SEAL subgraphs.

    Since subgraphs have different sizes, we need to batch them properly.
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

import torch
from torch_geometric.utils import negative_sampling, k_hop_subgraph
import torch.nn.functional as F


def k_hop_negative_sampling(edge_index, num_nodes, num_samples, k=2, device='cpu'):
    """
    Sample negative edges from k-hop neighborhoods.
    More informative than random sampling - samples from "close but not connected" nodes.

    Args:
        edge_index: Edge index tensor
        num_nodes: Number of nodes
        num_samples: Number of negative samples to generate
        k: Number of hops (2-3 recommended for hard negatives)
        device: Device to use

    Returns:
        torch.Tensor: Negative edges of shape (num_samples, 2)
    """
    # Build adjacency dict for efficient k-hop lookup
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    # Convert to sets for faster lookup
    adj_sets = [set(neighbors) for neighbors in adj_list]

    neg_edges = []
    attempts = 0
    max_attempts = num_samples * 50  # Prevent infinite loops

    while len(neg_edges) < num_samples and attempts < max_attempts:
        attempts += 1

        # Randomly select source node
        src = torch.randint(0, num_nodes, (1,)).item()

        # Get k-hop neighborhood (excluding 1-hop to ensure negatives)
        k_hop_nodes = set()
        current_level = {src}
        visited = {src}

        for hop in range(k):
            next_level = set()
            for node in current_level:
                neighbors = adj_sets[node]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        if hop == k - 1:  # Only add k-th hop nodes
                            k_hop_nodes.add(neighbor)
            current_level = next_level

        # Sample from k-hop neighborhood
        if len(k_hop_nodes) > 0:
            dst = list(k_hop_nodes)[torch.randint(0, len(k_hop_nodes), (1,)).item()]

            # Verify it's not an existing edge
            if dst not in adj_sets[src]:
                neg_edges.append([src, dst])

    # If we didn't get enough k-hop negatives, fill with random negatives
    if len(neg_edges) < num_samples:
        remaining = num_samples - len(neg_edges)
        random_neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=remaining
        ).t()
        neg_edges.extend(random_neg.tolist())

    return torch.tensor(neg_edges, dtype=torch.long, device=device)


def hard_negative_mining(model, z, edge_index, num_nodes, num_samples, top_k_ratio=0.3, device='cpu'):
    """
    Sample hard negatives - negatives with high predicted scores that are challenging for the model.

    Args:
        model: The model
        z: Node embeddings
        edge_index: Current edge index
        num_nodes: Number of nodes in the graph
        num_samples: Number of negatives to return
        top_k_ratio: Ratio of hard negatives to mine from a larger pool
        device: Device to put negatives on

    Returns:
        torch.Tensor: Hard negative edges of shape (num_samples, 2)
    """
    # Sample more negatives than needed
    sample_size = int(num_samples / top_k_ratio)
    neg_edges = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=sample_size
    ).t().to(device)

    # Score all negative samples
    with torch.no_grad():
        neg_scores = model.decode(z, neg_edges)

    # Select top-k hardest negatives (highest scores = closest to positive)
    _, indices = torch.topk(neg_scores, k=num_samples)
    hard_negatives = neg_edges[indices]

    return hard_negatives

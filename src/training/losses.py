import torch
from torch_geometric.utils import negative_sampling


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

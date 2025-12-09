import torch


def evaluate(model, data, evaluator, pos_edges, neg_edges, batch_size=50000):
    """
    Evaluate model using official OGB negative edges.

    Args:
        model: Model to evaluate
        data: Graph data object
        evaluator: OGB evaluator
        pos_edges: Positive edges for evaluation
        neg_edges: Negative edges for evaluation
        batch_size: Batch size for evaluation

    Returns:
        float: Hits@20 score
    """
    model.eval()
    with torch.no_grad():
        # Check if model needs input features (Morgan models) or uses embeddings
        if hasattr(data, 'x') and data.x is not None:
            z = model.encode(data.edge_index, x=data.x)
        else:
            z = model.encode(data.edge_index)

        # Positive scores - batch process to avoid OOM
        pos_scores_list = []
        for i in range(0, pos_edges.size(0), batch_size):
            chunk = pos_edges[i:i+batch_size]
            scores = model.decode(z, chunk).view(-1).cpu()
            pos_scores_list.append(scores)
            del scores
        pos_scores = torch.cat(pos_scores_list)
        del pos_scores_list

        # Negative scores - batch process to avoid OOM
        neg_scores_list = []
        for i in range(0, neg_edges.size(0), batch_size):
            chunk = neg_edges[i:i+batch_size]
            scores = model.decode(z, chunk).view(-1).cpu()
            neg_scores_list.append(scores)
            del scores
        neg_scores = torch.cat(neg_scores_list)
        del neg_scores_list

        # Free z before evaluation
        del z
        # Clear cache for CUDA and MPS devices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Use OGB evaluator
        result = evaluator.eval({
            'y_pred_pos': pos_scores,
            'y_pred_neg': neg_scores,
        })

        return result['hits@20']

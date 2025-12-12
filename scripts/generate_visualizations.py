"""
Generate visualizations for the GDINN paper/blog post.
Creates figures for:
1. Dataset statistics (degree distribution, split sizes)
2. Feature coverage and dimensionality
3. t-SNE of drug embeddings
4. Model performance comparison
5. Network graph visualization
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ogb.linkproppred import PygLinkPropPredDataset
from sklearn.manifold import TSNE
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = project_root / "docs" / "ddi_model" / "final_blog" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ogb_data():
    """Load the ogbl-ddi dataset."""
    print("Loading ogbl-ddi dataset...")
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', root=str(project_root / 'dataset'))
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    return data, split_edge, dataset


def plot_degree_distribution(data, save=True):
    """Plot the degree distribution of the graph."""
    print("Generating degree distribution plot...")

    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes

    # Count degrees
    degrees = np.zeros(num_nodes, dtype=int)
    np.add.at(degrees, edge_index[0], 1)
    np.add.at(degrees, edge_index[1], 1)
    degrees = degrees // 2  # Undirected graph, edges counted twice

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax1 = axes[0]
    ax1.hist(degrees, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Number of Drugs', fontsize=12)
    ax1.set_title('Degree Distribution', fontsize=14)
    ax1.axvline(np.mean(degrees), color='red', linestyle='--', label=f'Mean: {np.mean(degrees):.1f}')
    ax1.axvline(np.median(degrees), color='orange', linestyle='--', label=f'Median: {np.median(degrees):.1f}')
    ax1.legend()

    # Log-log plot
    ax2 = axes[1]
    degree_counts = Counter(degrees)
    degs = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degs]
    ax2.scatter(degs, counts, alpha=0.6, s=20, color='steelblue')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree (log scale)', fontsize=12)
    ax2.set_ylabel('Count (log scale)', fontsize=12)
    ax2.set_title('Degree Distribution (Log-Log)', fontsize=14)

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'degree_distribution.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'degree_distribution.png'}")

    plt.close()

    # Print statistics
    print(f"  Degree statistics:")
    print(f"    Min: {np.min(degrees)}, Max: {np.max(degrees)}")
    print(f"    Mean: {np.mean(degrees):.2f}, Median: {np.median(degrees):.2f}")
    print(f"    Std: {np.std(degrees):.2f}")

    return degrees


def plot_dataset_splits(split_edge, save=True):
    """Plot the dataset split sizes."""
    print("Generating dataset split visualization...")

    train_edges = split_edge['train']['edge'].shape[0]
    val_pos = split_edge['valid']['edge'].shape[0]
    val_neg = split_edge['valid']['edge_neg'].shape[0]
    test_pos = split_edge['test']['edge'].shape[0]
    test_neg = split_edge['test']['edge_neg'].shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart of edge counts
    ax1 = axes[0]
    categories = ['Train\n(positive)', 'Valid\n(positive)', 'Valid\n(negative)', 'Test\n(positive)', 'Test\n(negative)']
    counts = [train_edges, val_pos, val_neg, test_pos, test_neg]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#e67e22']

    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('Dataset Split Sizes', fontsize=14)
    ax1.set_ylim(0, max(counts) * 1.15)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000,
                f'{count:,}', ha='center', va='bottom', fontsize=10)

    # Pie chart
    ax2 = axes[1]
    sizes = [train_edges, val_pos + val_neg, test_pos + test_neg]
    labels = [f'Train\n({train_edges:,})', f'Valid\n({val_pos + val_neg:,})', f'Test\n({test_pos + test_neg:,})']
    colors_pie = ['#2ecc71', '#3498db', '#9b59b6']
    explode = (0.02, 0.02, 0.02)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax2.set_title('Edge Distribution by Split', fontsize=14)

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'dataset_splits.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'dataset_splits.png'}")

    plt.close()


def plot_feature_overview(save=True):
    """Plot feature dimensionality and coverage."""
    print("Generating feature overview visualization...")

    data_dir = project_root / 'data'

    # Feature dimensions
    features = {
        'Morgan FP': 2048,
        'ChemBERTa': 768,
        'Drug-Target': 229,
        'PubChem': 9,
    }

    # Try to load coverage information
    coverage = {}
    num_nodes = 4267

    # Check Morgan features
    morgan_path = data_dir / 'morgan_features_2048.pt'
    if morgan_path.exists():
        morgan = torch.load(morgan_path)
        # Count non-zero rows
        morgan_valid = (morgan.sum(dim=1) > 0).sum().item()
        coverage['Morgan FP'] = morgan_valid / num_nodes * 100

    # Check ChemBERTa features
    chemberta_mask_path = data_dir / 'chemberta_features_768_mask.pt'
    if chemberta_mask_path.exists():
        mask = torch.load(chemberta_mask_path)
        coverage['ChemBERTa'] = mask.sum().item() / num_nodes * 100

    # Assume full coverage for others
    coverage['Drug-Target'] = 100.0
    coverage['PubChem'] = 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Feature dimensions
    ax1 = axes[0]
    names = list(features.keys())
    dims = list(features.values())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    bars = ax1.barh(names, dims, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Dimensions', fontsize=12)
    ax1.set_title('External Feature Dimensions', fontsize=14)
    ax1.set_xlim(0, max(dims) * 1.15)

    for bar, dim in zip(bars, dims):
        ax1.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                f'{dim}', ha='left', va='center', fontsize=11, fontweight='bold')

    # Add total
    ax1.text(0.95, 0.05, f'Total: {sum(dims)} dims', transform=ax1.transAxes,
            ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Feature coverage
    ax2 = axes[1]
    cov_names = list(coverage.keys())
    cov_vals = list(coverage.values())

    bars2 = ax2.barh(cov_names, cov_vals, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Coverage (%)', fontsize=12)
    ax2.set_title('Feature Coverage (% of drugs)', fontsize=14)
    ax2.set_xlim(0, 110)

    for bar, cov in zip(bars2, cov_vals):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{cov:.1f}%', ha='left', va='center', fontsize=11)

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'feature_overview.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'feature_overview.png'}")

    plt.close()


def plot_model_comparison(save=True):
    """Plot model performance comparison."""
    print("Generating model comparison visualization...")

    # Results from the paper
    models = {
        'GraphSAGE\n(baseline)': {'val': 9.61, 'test': 4.46},
        'GCN\n(baseline)': {'val': 13.59, 'test': 11.02},
        'GAT\n(baseline)': {'val': 9.53, 'test': 4.88},
        'GraphTrans.\n(baseline)': {'val': 11.66, 'test': 3.73},
        'Advanced\nGCN': {'val': 64.78, 'test': 61.69},
        'GDINN': {'val': 70.31, 'test': 73.28},
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    val_scores = [models[m]['val'] for m in models]
    test_scores = [models[m]['test'] for m in models]

    bars1 = ax.bar(x - width/2, val_scores, width, label='Validation Hits@20',
                   color='#3498db', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test Hits@20',
                   color='#e74c3c', edgecolor='black', alpha=0.8)

    ax.set_ylabel('Hits@20 (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models.keys(), fontsize=10)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 85)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
               ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
               ha='center', va='bottom', fontsize=9)

    # Add vertical line separating baselines from advanced models
    ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.5, 80, 'Simple Baselines', ha='center', fontsize=11, style='italic', color='gray')
    ax.text(4.5, 80, 'Our Models', ha='center', fontsize=11, style='italic', color='gray')

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'model_comparison.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'model_comparison.png'}")

    plt.close()


def plot_feature_ablation(save=True):
    """Plot feature ablation results."""
    print("Generating feature ablation visualization...")

    # Results from the paper
    configs = {
        'Advanced GCN\n(no features)': {'val': 64.78, 'test': 61.69},
        '+ Morgan': {'val': 70.45, 'test': 55.05},
        '+ PubChem': {'val': 69.92, 'test': 56.72},
        '+ ChemBERTa': {'val': 70.85, 'test': 59.01},
        '+ Drug-Target': {'val': 70.49, 'test': 54.18},
        'GDINN\n(all features)': {'val': 70.31, 'test': 73.28},
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    width = 0.35

    val_scores = [configs[m]['val'] for m in configs]
    test_scores = [configs[m]['test'] for m in configs]

    bars1 = ax.bar(x - width/2, val_scores, width, label='Validation Hits@20',
                   color='#3498db', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test Hits@20',
                   color='#e74c3c', edgecolor='black', alpha=0.8)

    ax.set_ylabel('Hits@20 (%)', fontsize=12)
    ax.set_title('Feature Ablation Study', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs.keys(), fontsize=10)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 85)

    # Add horizontal line for baseline test performance
    ax.axhline(y=61.69, color='gray', linestyle=':', alpha=0.7, label='Baseline test')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
               ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}',
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'feature_ablation.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'feature_ablation.png'}")

    plt.close()


def plot_tsne_embeddings(data, degrees, save=True):
    """Plot t-SNE of drug features colored by degree."""
    print("Generating t-SNE visualization...")

    data_dir = project_root / 'data'

    # Try to load Morgan fingerprints for visualization
    morgan_path = data_dir / 'morgan_features_2048.pt'
    if not morgan_path.exists():
        print("  Morgan features not found, skipping t-SNE")
        return

    morgan = torch.load(morgan_path).numpy()

    # Subsample for speed (t-SNE is slow)
    n_samples = min(2000, morgan.shape[0])
    indices = np.random.choice(morgan.shape[0], n_samples, replace=False)
    morgan_subset = morgan[indices]
    degrees_subset = degrees[indices]

    print(f"  Running t-SNE on {n_samples} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(morgan_subset)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=degrees_subset, cmap='viridis', alpha=0.6, s=20)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Node Degree', fontsize=12)

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE of Drug Morgan Fingerprints\n(colored by degree)', fontsize=14)

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'tsne_embeddings.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'tsne_embeddings.png'}")

    plt.close()


def plot_graph_stats_summary(data, split_edge, save=True):
    """Create a summary statistics figure."""
    print("Generating graph statistics summary...")

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2  # Undirected
    density = 2 * num_edges / (num_nodes * (num_nodes - 1)) * 100

    train_edges = split_edge['train']['edge'].shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    stats_text = f"""
    ogbl-ddi Dataset Statistics
    {'='*40}

    Graph Properties:
      • Nodes (drugs):     {num_nodes:,}
      • Edges (interactions): {num_edges:,}
      • Edge density:      {density:.2f}%
      • Avg. degree:       ~{2*num_edges/num_nodes:.0f}

    Dataset Splits:
      • Training edges:    {train_edges:,}
      • Validation edges:  {split_edge['valid']['edge'].shape[0]:,} (+) / {split_edge['valid']['edge_neg'].shape[0]:,} (-)
      • Test edges:        {split_edge['test']['edge'].shape[0]:,} (+) / {split_edge['test']['edge_neg'].shape[0]:,} (-)

    External Features:
      • Morgan fingerprints: 2,048 dims
      • ChemBERTa embeddings: 768 dims
      • Drug-target vectors:  229 dims
      • PubChem properties:   9 dims
      • Total:               3,054 dims
    """

    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='center', horizontalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'dataset_summary.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'dataset_summary.png'}")

    plt.close()


def load_drug_names():
    """Load drug name mappings from ogbl-ddi."""
    drug_mapping = pd.read_csv(project_root / 'dataset/ogbl_ddi/mapping/nodeidx2drugid.csv.gz')
    ddi_desc = pd.read_csv(project_root / 'dataset/ogbl_ddi/mapping/ddi_description.csv.gz')

    drug_names = {}
    for _, row in ddi_desc.iterrows():
        drug_names[row['first drug id']] = row['first drug name']
        drug_names[row['second drug id']] = row['second drug name']

    idx_to_name = {}
    name_to_idx = {}
    for _, row in drug_mapping.iterrows():
        drug_id = row['drug id']
        if drug_id in drug_names:
            idx_to_name[row['node idx']] = drug_names[drug_id]
            name_to_idx[drug_names[drug_id]] = row['node idx']

    return idx_to_name, name_to_idx


def plot_labeled_network(data, save=True):
    """Plot a network with labeled drug names showing specific interactions."""
    print("Generating labeled network visualization...")

    idx_to_name, name_to_idx = load_drug_names()
    edge_index = data.edge_index.numpy()

    # Build adjacency
    adj = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)

    # Well-known drugs with many interconnections (9/10 edges between them)
    selected_drugs = ['Ibuprofen', 'Metformin', 'Tramadol', 'Fluoxetine', 'Sertraline']

    nodes = []
    for name in selected_drugs:
        if name in name_to_idx:
            nodes.append((name_to_idx[name], name))

    print(f"  Selected drugs: {[n[1] for n in nodes]}")

    # Build graph with just these drugs and their connections
    G = nx.Graph()
    node_idxs = set(n[0] for n in nodes)

    for idx, name in nodes:
        G.add_node(idx, label=name)

    # Add edges between selected drugs
    for i, (idx1, name1) in enumerate(nodes):
        for idx2, name2 in nodes[i+1:]:
            if idx2 in adj.get(idx1, set()):
                G.add_edge(idx1, idx2)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    fig, ax = plt.subplots(figsize=(15, 11))
    pos = nx.spring_layout(G, k=1, iterations=100, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.6, width=4, edge_color='#555555', ax=ax)

    nx.draw_networkx_nodes(G, pos, node_size=15000, node_color='#5dade2',
                           edgecolors='#2471a3', linewidths=3, ax=ax)

    labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=23, font_weight='bold', ax=ax)

    ax.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / 'network_example.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'network_example.png'}")
    plt.close()


def plot_single_drug_network(data, save=True):
    """Plot interactions for a single drug (Warfarin) in radial layout."""
    print("Generating single-drug network visualization...")

    idx_to_name, name_to_idx = load_drug_names()
    edge_index = data.edge_index.numpy()

    adj = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj.setdefault(src, set()).add(dst)

    # Find Warfarin
    target_drug = None
    for name in name_to_idx:
        if 'warfarin' in name.lower():
            target_drug = name
            break

    if target_drug is None:
        print("  Warfarin not found, skipping")
        return

    target_idx = name_to_idx[target_drug]
    neighbors = list(adj.get(target_idx, []))
    print(f"  {target_drug}: {len(neighbors)} interactions")

    named_neighbors = [(n, idx_to_name[n]) for n in neighbors if n in idx_to_name]
    named_neighbors = sorted(named_neighbors, key=lambda x: len(x[1]))[:15]

    G = nx.Graph()
    G.add_node(target_idx, label=target_drug, is_center=True)
    for n_idx, n_name in named_neighbors:
        G.add_node(n_idx, label=n_name, is_center=False)
        G.add_edge(target_idx, n_idx)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Radial layout
    pos = {target_idx: np.array([0, 0])}
    n = len(named_neighbors)
    for i, (n_idx, _) in enumerate(named_neighbors):
        angle = 2 * np.pi * i / n
        pos[n_idx] = np.array([np.cos(angle), np.sin(angle)]) * 1.2

    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, edge_color='#3498db', ax=ax)

    outer_nodes = [n for n, _ in named_neighbors]
    nx.draw_networkx_nodes(G, pos, nodelist=outer_nodes, node_size=1200,
                           node_color='#a8d5e5', edgecolors='#2980b9', linewidths=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[target_idx], node_size=2500,
                           node_color='#e74c3c', edgecolors='#c0392b', linewidths=3, ax=ax)

    for n_idx, n_name in named_neighbors:
        x, y = pos[n_idx]
        angle = np.arctan2(y, x) * 180 / np.pi
        offset = 0.15
        label_x, label_y = x * (1 + offset), y * (1 + offset)
        ha = 'right' if abs(angle) > 90 else 'left'
        if abs(angle) > 90:
            angle += 180
        ax.text(label_x, label_y, n_name, fontsize=10, fontweight='bold',
                ha=ha, va='center', rotation=angle, rotation_mode='anchor')

    ax.text(0, 0, target_drug, fontsize=14, fontweight='bold', ha='center', va='center', color='white')
    ax.set_title(f'Drug Interactions of {target_drug}\n({len(neighbors)} total, showing {len(named_neighbors)})', fontsize=14)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / 'network_single_drug.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'network_single_drug.png'}")
    plt.close()


def plot_network_graph(data, degrees, save=True):
    """Plot a subgraph visualization using NetworkX."""
    print("Generating network graph visualization...")

    edge_index = data.edge_index.numpy()

    # Create NetworkX graph from a subset (full graph is too large)
    # Select high-degree nodes for a more interesting visualization
    top_k = 100  # Number of nodes to visualize
    top_nodes = np.argsort(degrees)[-top_k:]

    # Create subgraph
    G = nx.Graph()
    G.add_nodes_from(top_nodes)

    # Add edges between selected nodes
    edge_set = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src in top_nodes and dst in top_nodes:
            if src < dst:  # Avoid duplicates for undirected
                edge_set.add((src, dst))

    G.add_edges_from(edge_set)

    print(f"  Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Node sizes based on degree
    node_sizes = [degrees[n] / 5 for n in G.nodes()]

    # Node colors based on degree
    node_colors = [degrees[n] for n in G.nodes()]

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color=node_colors, cmap='viridis',
                                   alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

    # Colorbar
    cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
    cbar.set_label('Node Degree', fontsize=12)

    ax.set_title(f'Drug-Drug Interaction Network\n(Top {top_k} highest-degree drugs)', fontsize=14)
    ax.axis('off')

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'network_graph.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'network_graph.png'}")

    plt.close()


def plot_network_graph_sample(data, degrees, save=True):
    """Plot a random sample subgraph for better visual clarity."""
    print("Generating sampled network visualization...")

    edge_index = data.edge_index.numpy()

    # Sample random nodes and their neighbors
    np.random.seed(42)
    seed_nodes = np.random.choice(data.num_nodes, size=20, replace=False)

    # Get 1-hop neighbors
    neighbors = set(seed_nodes)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src in seed_nodes:
            neighbors.add(dst)
        if dst in seed_nodes:
            neighbors.add(src)

    # Limit to manageable size
    neighbors = list(neighbors)[:200]
    neighbor_set = set(neighbors)

    # Create subgraph
    G = nx.Graph()
    G.add_nodes_from(neighbors)

    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src in neighbor_set and dst in neighbor_set:
            G.add_edge(src, dst)

    print(f"  Sampled subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Highlight seed nodes
    node_colors = ['#e74c3c' if n in seed_nodes else '#3498db' for n in G.nodes()]
    node_sizes = [300 if n in seed_nodes else 50 for n in G.nodes()]

    # Layout
    pos = nx.kamada_kawai_layout(G)

    # Draw edges first
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.3, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color=node_colors, alpha=0.8, ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=12, label='Seed drugs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=8, label='Neighboring drugs'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.set_title(f'Drug-Drug Interaction Network (Sampled Subgraph)\n{G.number_of_nodes()} drugs, {G.number_of_edges()} interactions',
                fontsize=14)
    ax.axis('off')

    plt.tight_layout()

    if save:
        plt.savefig(OUTPUT_DIR / 'network_sample.png', bbox_inches='tight', dpi=150)
        print(f"  Saved to {OUTPUT_DIR / 'network_sample.png'}")

    plt.close()


def main():
    print("=" * 60)
    print("Generating visualizations for GDINN paper")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load data
    data, split_edge, dataset = load_ogb_data()

    # Generate all visualizations
    degrees = plot_degree_distribution(data)
    # plot_dataset_splits(split_edge)
    # plot_feature_overview()
    # plot_model_comparison()
    # plot_feature_ablation()
    # plot_tsne_embeddings(data, degrees)
    # plot_graph_stats_summary(data, split_edge)
    plot_labeled_network(data)
    # plot_single_drug_network(data)

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output files are in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

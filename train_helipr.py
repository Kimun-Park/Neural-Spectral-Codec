#!/usr/bin/env python3
"""
Train on HeLiPR (hard dataset) and evaluate cross-dataset generalization.

Hypothesis: Training on structurally uniform environment might force the model
to learn more discriminative features.
"""

import sys
sys.path.insert(0, '/workspace/Neural-Spectral-Codec/src')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from scipy.stats import wasserstein_distance
from pathlib import Path
import time

# Import modules
from data.helipr_loader import HeLiPRLoader
from data.kitti_loader import KITTILoader
from encoding.spectral_encoder import SpectralEncoder
from gnn.model import SpectralGNN


def load_helipr_data(sequence_path, n_samples=None):
    """Load HeLiPR data"""
    print(f"Loading {sequence_path}...")
    loader = HeLiPRLoader(sequence_path, lazy_load=True)

    if n_samples and n_samples < len(loader):
        indices = np.linspace(0, len(loader)-1, n_samples).astype(int)
    else:
        indices = range(len(loader))

    scans = [loader._load_velodyne(loader.scan_files[i]) for i in indices]
    poses = [loader.scan_poses[i] for i in indices]

    print(f"  Loaded {len(scans)} scans")
    return scans, poses


def load_kitti_data(sequence, n_samples=None):
    """Load KITTI data"""
    print(f"Loading KITTI {sequence}...")
    loader = KITTILoader('/workspace/data/kitti/dataset', sequence, lazy_load=True)

    if n_samples and n_samples < len(loader):
        indices = np.linspace(0, len(loader)-1, n_samples).astype(int)
    else:
        indices = range(len(loader))

    scans = [loader._load_point_cloud(i) for i in indices]
    poses = [loader.poses[i] for i in indices]

    print(f"  Loaded {len(scans)} scans")
    return scans, poses


def encode_scans(scans, encoder):
    """Encode scans to spectral histograms"""
    histograms = []
    for i, scan in enumerate(scans):
        if i % 500 == 0:
            print(f"  Encoding {i}/{len(scans)}...")
        hist = encoder.encode_points(scan)
        if isinstance(hist, torch.Tensor):
            hist = hist.detach().cpu().numpy()
        histograms.append(hist)
    return np.array(histograms)


def build_graph(histograms, poses, temporal_neighbors=5):
    """Build temporal graph for GNN"""
    n = len(histograms)
    positions = np.array([p[:3, 3] for p in poses])

    # Create edges (temporal neighbors)
    edge_index = []
    edge_attr = []

    for i in range(n):
        for j in range(max(0, i - temporal_neighbors), min(n, i + temporal_neighbors + 1)):
            if i != j:
                edge_index.append([i, j])
                dist = np.linalg.norm(positions[i] - positions[j])
                edge_attr.append([dist])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(histograms, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), positions


def mine_triplets(positions, n_triplets=1000, pos_thresh=5.0, neg_min=10.0, neg_max=50.0, skip=30):
    """Mine triplets for training"""
    n = len(positions)
    triplets = []

    np.random.seed(42)
    attempts = 0
    max_attempts = n_triplets * 100

    while len(triplets) < n_triplets and attempts < max_attempts:
        attempts += 1
        anchor = np.random.randint(0, n)

        # Find positive (close in space, far in time)
        pos_candidates = []
        for i in range(n):
            if abs(i - anchor) > skip:
                dist = np.linalg.norm(positions[anchor] - positions[i])
                if dist < pos_thresh:
                    pos_candidates.append(i)

        if not pos_candidates:
            continue

        positive = np.random.choice(pos_candidates)

        # Find negative (far in space)
        neg_candidates = []
        for i in range(n):
            dist = np.linalg.norm(positions[anchor] - positions[i])
            if neg_min < dist < neg_max:
                neg_candidates.append(i)

        if not neg_candidates:
            continue

        negative = np.random.choice(neg_candidates)
        triplets.append((anchor, positive, negative))

    print(f"  Mined {len(triplets)} triplets")
    return triplets


def train_gnn(graph, triplets, positions, n_epochs=50, lr=0.001, margin=0.1):
    """Train GNN with triplet loss"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    model = SpectralGNN(
        input_dim=graph.x.shape[1],
        hidden_dim=graph.x.shape[1],
        output_dim=graph.x.shape[1],
        n_layers=3,
        edge_dim=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    graph = graph.to(device)

    best_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()

        # Forward pass
        embeddings = model(graph)

        # Compute triplet loss
        total_loss = 0
        n_batches = 0
        batch_size = 64

        for i in range(0, len(triplets), batch_size):
            batch_triplets = triplets[i:i+batch_size]

            anchors = torch.stack([embeddings[t[0]] for t in batch_triplets])
            positives = torch.stack([embeddings[t[1]] for t in batch_triplets])
            negatives = torch.stack([embeddings[t[2]] for t in batch_triplets])

            loss = triplet_loss_fn(anchors, positives, negatives)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Backward pass
        optimizer.zero_grad()

        # Recompute for gradient
        embeddings = model(graph)
        batch_triplets = triplets[:batch_size]
        anchors = torch.stack([embeddings[t[0]] for t in batch_triplets])
        positives = torch.stack([embeddings[t[1]] for t in batch_triplets])
        negatives = torch.stack([embeddings[t[2]] for t in batch_triplets])
        loss = triplet_loss_fn(anchors, positives, negatives)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    return model


def evaluate(histograms, positions, method='wasserstein', model=None, graph=None, device=None):
    """Evaluate retrieval performance"""
    n = len(histograms)
    skip = 30

    # Get embeddings if using GNN
    if method == 'gnn' and model is not None:
        model.eval()
        with torch.no_grad():
            embeddings = model(graph.to(device)).cpu().numpy()
    else:
        embeddings = histograms

    # Find loop closure queries
    queries = []
    for i in range(n):
        for j in range(i + skip, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 5.0:
                queries.append((i, j))
                break  # One positive per query

    if len(queries) == 0:
        return {'r@1': 0, 'r@5': 0, 'r@10': 0}

    # Evaluate
    correct_at_k = {1: 0, 5: 0, 10: 0}

    for query_idx, true_match in queries[:500]:  # Limit for speed
        # Compute distances to all candidates
        candidates = []
        for i in range(n):
            if abs(i - query_idx) > skip:
                if method == 'wasserstein':
                    d = wasserstein_distance(embeddings[query_idx], embeddings[i])
                else:  # L2
                    d = np.linalg.norm(embeddings[query_idx] - embeddings[i])
                candidates.append((i, d))

        # Sort by distance
        candidates.sort(key=lambda x: x[1])

        # Check if true match is in top-k
        for k in [1, 5, 10]:
            top_k_indices = [c[0] for c in candidates[:k]]
            # Check if any top-k is within 5m of query
            for idx in top_k_indices:
                if np.linalg.norm(positions[query_idx] - positions[idx]) < 5.0:
                    correct_at_k[k] += 1
                    break

    results = {
        'r@1': correct_at_k[1] / len(queries[:500]) * 100,
        'r@5': correct_at_k[5] / len(queries[:500]) * 100,
        'r@10': correct_at_k[10] / len(queries[:500]) * 100,
        'n_queries': len(queries[:500])
    }

    return results


def main():
    print("="*70)
    print("Training on Hard Dataset (HeLiPR Town01)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Encoder for VLP-16
    encoder_vlp16 = SpectralEncoder(
        n_elevation=16, n_azimuth=360, n_bins=100,  # Using 100 bins
        elevation_range=(-15.0, 15.0),
        interpolate_empty=True, learnable_alpha=False
    )

    # Encoder for HDL-64E
    encoder_hdl64 = SpectralEncoder(
        n_elevation=64, n_azimuth=360, n_bins=100,
        elevation_range=(-24.8, 2.0),
        interpolate_empty=True, learnable_alpha=False
    )

    # ========== Load Training Data (Town01) ==========
    print("\n" + "="*70)
    print("Loading Training Data: HeLiPR Town01")
    print("="*70)

    train_scans, train_poses = load_helipr_data(
        '/workspace/data/helipr/Town01/Town01',
        n_samples=3000  # Use 3000 samples for training
    )

    print("\nEncoding training data...")
    train_hists = encode_scans(train_scans, encoder_vlp16)

    print("\nBuilding graph...")
    train_graph, train_positions = build_graph(train_hists, train_poses)

    print("\nMining triplets...")
    triplets = mine_triplets(train_positions, n_triplets=2000)

    # ========== Train GNN ==========
    print("\n" + "="*70)
    print("Training GNN")
    print("="*70)

    model = train_gnn(train_graph, triplets, train_positions, n_epochs=50)

    # ========== Evaluate on Town01 (train set) ==========
    print("\n" + "="*70)
    print("Evaluating on Town01 (train set)")
    print("="*70)

    print("\nSpectral Histogram (Wasserstein):")
    results_hist = evaluate(train_hists, train_positions, method='wasserstein')
    print(f"  R@1: {results_hist['r@1']:.2f}%, R@5: {results_hist['r@5']:.2f}%, R@10: {results_hist['r@10']:.2f}%")

    print("\nGNN Embeddings (L2):")
    results_gnn = evaluate(train_hists, train_positions, method='gnn',
                          model=model, graph=train_graph, device=device)
    print(f"  R@1: {results_gnn['r@1']:.2f}%, R@5: {results_gnn['r@5']:.2f}%, R@10: {results_gnn['r@10']:.2f}%")

    # ========== Evaluate on Roundabout01 (unseen) ==========
    print("\n" + "="*70)
    print("Evaluating on Roundabout01 (unseen)")
    print("="*70)

    test_scans, test_poses = load_helipr_data(
        '/workspace/data/helipr/Roundabout01/Roundabout01',
        n_samples=2000
    )

    print("\nEncoding test data...")
    test_hists = encode_scans(test_scans, encoder_vlp16)
    test_graph, test_positions = build_graph(test_hists, test_poses)

    print("\nSpectral Histogram (Wasserstein):")
    results_hist = evaluate(test_hists, test_positions, method='wasserstein')
    print(f"  R@1: {results_hist['r@1']:.2f}%, R@5: {results_hist['r@5']:.2f}%, R@10: {results_hist['r@10']:.2f}%")

    # ========== Evaluate on KITTI 09 (cross-sensor) ==========
    print("\n" + "="*70)
    print("Evaluating on KITTI 09 (cross-sensor, unseen)")
    print("="*70)

    kitti_scans, kitti_poses = load_kitti_data('09', n_samples=1500)

    print("\nEncoding KITTI data...")
    kitti_hists = encode_scans(kitti_scans, encoder_hdl64)
    kitti_graph, kitti_positions = build_graph(kitti_hists, kitti_poses)

    print("\nSpectral Histogram (Wasserstein):")
    results_hist = evaluate(kitti_hists, kitti_positions, method='wasserstein')
    print(f"  R@1: {results_hist['r@1']:.2f}%, R@5: {results_hist['r@5']:.2f}%, R@10: {results_hist['r@10']:.2f}%")

    # ========== Summary ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nTraining: HeLiPR Town01 (hard, structurally uniform)")
    print("\n| Dataset | R@1 |")
    print("|---------|-----|")
    print(f"| Town01 (train) | {results_gnn['r@1']:.1f}% |")
    print(f"| Roundabout01 (unseen) | - |")
    print(f"| KITTI 09 (cross-sensor) | {results_hist['r@1']:.1f}% |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze if GNN is helping or hurting loop closure performance
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import yaml
from scipy.spatial.distance import cdist

from data.kitti_loader import KITTILoader
from encoding.spectral_encoder import SpectralEncoder
from gnn.model import SpectralGNN
from keyframe.selector import KeyframeSelector
from keyframe.graph_manager import build_graph_from_keyframes_batch


def load_model(checkpoint_path, config):
    """Load trained model"""
    model = SpectralGNN(
        input_dim=config['gnn']['input_dim'],
        hidden_dim=config['gnn']['hidden_dim'],
        output_dim=config['gnn']['output_dim'],
        n_layers=config['gnn']['n_layers'],
        edge_dim=1
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Handle "gnn." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('gnn.'):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model


def compute_loop_closure_recall(embeddings, poses, skip_frames=30, distance_threshold=5.0):
    """Compute loop closure recall"""
    n = len(embeddings)
    positions = poses[:, :3, 3]

    # Compute pairwise distances
    embedding_distances = cdist(embeddings, embeddings, metric='euclidean')
    pose_distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=2
    )

    # Find loop closure queries
    queries = []
    for i in range(n):
        for j in range(i + skip_frames, n):
            if pose_distances[i, j] < distance_threshold:
                queries.append((j, i))
                break

    if len(queries) == 0:
        return {1: 0, 5: 0, 10: 0}, 0

    results = {1: 0, 5: 0, 10: 0}

    for query_idx, true_match in queries:
        candidates = []
        for i in range(n):
            if abs(i - query_idx) > skip_frames:
                candidates.append((i, embedding_distances[query_idx, i], pose_distances[query_idx, i]))

        candidates.sort(key=lambda x: x[1])

        for k in [1, 5, 10]:
            for idx, emb_dist, geo_dist in candidates[:k]:
                if geo_dist < distance_threshold:
                    results[k] += 1
                    break

    for k in results:
        results[k] = results[k] / len(queries) * 100

    return results, len(queries)


def main():
    print("=" * 70)
    print("GNN Effect Analysis: Is GNN Helping or Hurting?")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('configs/training_helipr_to_kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    print("\nLoading model...")
    model = load_model('src/checkpoints/best_model.pth', config)
    model.to(device)
    model.eval()

    # Create encoder for KITTI (HDL-64E)
    encoder = SpectralEncoder(
        n_elevation=64,
        n_azimuth=360,
        elevation_range=(-24.8, 2.0),
        target_elevation_bins=16,
        n_bins=config['encoding']['n_bins'],
        interpolate_empty=True,
        learnable_alpha=False
    )

    keyframe_selector = KeyframeSelector(
        distance_threshold=config['keyframe']['distance_threshold'],
        rotation_threshold=config['keyframe']['rotation_threshold']
    )

    # Load KITTI 09
    print("\nLoading KITTI 09...")
    loader = KITTILoader('/workspace/data/kitti/dataset', '09', lazy_load=True)

    keyframes = []
    keyframe_selector.reset()

    for i in range(len(loader)):
        sample = loader[i]
        points = sample['points']
        pose = sample['pose']
        timestamp = sample.get('timestamp', float(i))

        descriptor = encoder.encode_points(points)
        if isinstance(descriptor, torch.Tensor):
            descriptor = descriptor.detach().cpu().numpy()

        is_keyframe, keyframe_obj, _ = keyframe_selector.process_scan(
            scan_id=i,
            points=points,
            pose=pose,
            timestamp=timestamp,
            force_first=(i == 0)
        )

        if is_keyframe and keyframe_obj is not None:
            keyframe_obj.descriptor = descriptor
            keyframes.append(keyframe_obj)

    print(f"Keyframes: {len(keyframes)}")

    # Extract data
    descriptors = np.array([kf.descriptor for kf in keyframes])
    poses = np.array([kf.pose for kf in keyframes])

    print("\n" + "=" * 70)
    print("1. Descriptor Only (No GNN)")
    print("=" * 70)

    results_desc, n_queries = compute_loop_closure_recall(descriptors, poses)
    print(f"Loop closure queries: {n_queries}")
    print(f"R@1:  {results_desc[1]:.2f}%")
    print(f"R@5:  {results_desc[5]:.2f}%")
    print(f"R@10: {results_desc[10]:.2f}%")

    print("\n" + "=" * 70)
    print("2. With GNN (Normalized edge_attr - correct)")
    print("=" * 70)

    graph = build_graph_from_keyframes_batch(
        keyframes,
        temporal_neighbors=config['keyframe'].get('temporal_neighbors', 5),
        device=device,
        poses=poses
    )

    with torch.no_grad():
        embeddings = model(graph.to(device)).cpu().numpy()

    results_gnn, _ = compute_loop_closure_recall(embeddings, poses)
    print(f"R@1:  {results_gnn[1]:.2f}%")
    print(f"R@5:  {results_gnn[5]:.2f}%")
    print(f"R@10: {results_gnn[10]:.2f}%")

    print("\n" + "=" * 70)
    print("3. Comparison")
    print("=" * 70)
    print(f"{'Metric':<10} {'Descriptor':<15} {'GNN':<15} {'Change':<15}")
    print("-" * 55)

    for k in [1, 5, 10]:
        change = results_gnn[k] - results_desc[k]
        sign = "+" if change >= 0 else ""
        print(f"R@{k:<8} {results_desc[k]:<14.2f}% {results_gnn[k]:<14.2f}% {sign}{change:.2f}%")

    print("\n" + "=" * 70)
    print("4. Analysis: What is GNN learning?")
    print("=" * 70)

    # Check embedding similarity patterns
    emb_distances = cdist(embeddings, embeddings, metric='euclidean')
    desc_distances = cdist(descriptors, descriptors, metric='euclidean')

    n = len(keyframes)
    positions = poses[:, :3, 3]
    pose_distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=2
    )

    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)

    # Correlation between temporal offset and embedding similarity
    temporal_offsets = np.abs(np.arange(n)[:, np.newaxis] - np.arange(n)[np.newaxis, :])

    print("\nCorrelation Analysis:")
    from scipy.stats import pearsonr

    # Flatten for correlation
    temp_flat = temporal_offsets[mask]
    pose_flat = pose_distances[mask]
    emb_flat = emb_distances[mask]
    desc_flat = desc_distances[mask]

    # Sample for speed (correlation on full matrix is slow)
    sample_idx = np.random.choice(len(temp_flat), min(100000, len(temp_flat)), replace=False)

    corr_temp_emb, _ = pearsonr(temp_flat[sample_idx], emb_flat[sample_idx])
    corr_temp_desc, _ = pearsonr(temp_flat[sample_idx], desc_flat[sample_idx])
    corr_pose_emb, _ = pearsonr(pose_flat[sample_idx], emb_flat[sample_idx])
    corr_pose_desc, _ = pearsonr(pose_flat[sample_idx], desc_flat[sample_idx])

    print(f"  Temporal offset ↔ Descriptor distance: {corr_temp_desc:.3f}")
    print(f"  Temporal offset ↔ Embedding distance:  {corr_temp_emb:.3f}")
    print(f"  Spatial distance ↔ Descriptor distance: {corr_pose_desc:.3f}")
    print(f"  Spatial distance ↔ Embedding distance:  {corr_pose_emb:.3f}")

    print("""
이상적인 상황:
- Temporal offset ↔ Embedding distance: 0에 가까움 (시간과 무관)
- Spatial distance ↔ Embedding distance: 높음 (공간 유사성 반영)

실제:
- GNN이 temporal offset correlation을 더 강화하면 → loop closure 성능 저하
- GNN이 spatial distance correlation을 강화하면 → loop closure 성능 향상
""")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Debug script to compare training validation vs evaluation methodology
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


def compute_recall_training_style(embeddings, poses, k, distance_threshold=5.0):
    """
    Training validation style: all-vs-all, no skip_frames
    """
    n = len(embeddings)
    embedding_distances = cdist(embeddings, embeddings, metric='euclidean')

    # Compute pairwise pose distances
    positions = poses[:, :3, 3]
    pose_distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=2
    )

    # Exclude self-matches
    np.fill_diagonal(embedding_distances, np.inf)

    # Get top-K indices for all queries
    top_k_indices = np.argpartition(embedding_distances, min(k, n - 1), axis=1)[:, :k]

    # Check if any top-K match is within distance threshold
    row_indices = np.arange(n)[:, np.newaxis]
    top_k_pose_distances = pose_distances[row_indices, top_k_indices]

    recalls = (top_k_pose_distances <= distance_threshold).any(axis=1).astype(float)

    return recalls.mean() * 100


def compute_recall_evaluation_style(embeddings, poses, k, distance_threshold=5.0, skip_frames=30):
    """
    Evaluation style: only loop closure queries, with skip_frames
    """
    n = len(embeddings)
    positions = poses[:, :3, 3]

    # Find loop closure queries (revisits)
    queries = []
    for i in range(n):
        for j in range(i + skip_frames, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < distance_threshold:
                queries.append((j, i))  # query j, true match i
                break

    if len(queries) == 0:
        return 0.0, 0

    # Compute embedding distances
    embedding_distances = cdist(embeddings, embeddings, metric='euclidean')
    pose_distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=2
    )

    correct_at_k = 0

    for query_idx, true_match in queries:
        candidates = []
        for i in range(n):
            if abs(i - query_idx) > skip_frames:
                emb_dist = embedding_distances[query_idx, i]
                geo_dist = pose_distances[query_idx, i]
                candidates.append((i, emb_dist, geo_dist))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1])

        top_k = candidates[:k]
        for idx, emb_dist, geo_dist in top_k:
            if geo_dist < distance_threshold:
                correct_at_k += 1
                break

    return correct_at_k / len(queries) * 100, len(queries)


def main():
    print("=" * 70)
    print("Debug: Training Validation vs Evaluation Methodology")
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

    # Build graph (training style - with normalized edge_attr)
    print("\nBuilding graph (training style with normalized edge_attr)...")
    graph = build_graph_from_keyframes_batch(
        keyframes,
        temporal_neighbors=config['keyframe'].get('temporal_neighbors', 5),
        device=device,
        poses=poses
    )

    # Get embeddings
    with torch.no_grad():
        embeddings = model(graph.to(device)).cpu().numpy()

    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)

    # Compute both ways
    for k in [1, 5, 10]:
        train_r = compute_recall_training_style(embeddings, poses, k)
        eval_r, n_queries = compute_recall_evaluation_style(embeddings, poses, k)

        print(f"\nR@{k}:")
        print(f"  Training validation style (all-vs-all):     {train_r:.2f}%")
        print(f"  Evaluation style (loop closure, n={n_queries}): {eval_r:.2f}%")

    # Now test with raw edge_attr (evaluation bug)
    print("\n" + "=" * 70)
    print("Testing edge_attr normalization impact")
    print("=" * 70)

    # Build graph with RAW edge distances (like evaluate_helipr.py does)
    n = len(keyframes)
    edge_index = []
    edge_attr_raw = []
    positions = poses[:, :3, 3]
    temporal_neighbors = config['keyframe'].get('temporal_neighbors', 5)

    for i in range(n):
        for j in range(max(0, i - temporal_neighbors), min(n, i + temporal_neighbors + 1)):
            if i != j:
                edge_index.append([i, j])
                dist = np.linalg.norm(positions[i] - positions[j])
                edge_attr_raw.append([dist])  # RAW, not normalized!

    from torch_geometric.data import Data
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr_raw_t = torch.tensor(edge_attr_raw, dtype=torch.float)
    x = torch.tensor(descriptors, dtype=torch.float)

    graph_raw = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_raw_t).to(device)

    with torch.no_grad():
        embeddings_raw = model(graph_raw).cpu().numpy()

    print("\nWith RAW edge_attr (evaluation bug):")
    for k in [1, 5, 10]:
        train_r = compute_recall_training_style(embeddings_raw, poses, k)
        eval_r, n_queries = compute_recall_evaluation_style(embeddings_raw, poses, k)
        print(f"  R@{k}: train_style={train_r:.2f}%, eval_style={eval_r:.2f}%")

    print("\nWith NORMALIZED edge_attr (correct):")
    for k in [1, 5, 10]:
        train_r = compute_recall_training_style(embeddings, poses, k)
        eval_r, n_queries = compute_recall_evaluation_style(embeddings, poses, k)
        print(f"  R@{k}: train_style={train_r:.2f}%, eval_style={eval_r:.2f}%")

    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)
    print("""
문제점:
1. 학습 validation은 "all-vs-all" 방식 (모든 키프레임이 쿼리)
   - 시간적으로 가까운 프레임도 후보에 포함
   - 시간적 인접 프레임은 대부분 5m 이내 → 높은 recall

2. 평가 스크립트는 "loop closure" 방식 (재방문만 쿼리)
   - skip_frames=30으로 시간적 인접 프레임 제외
   - 실제 장소 재인식 능력만 측정 → 낮은 recall

3. edge_attr 정규화 불일치
   - 학습: log(1 + dist) / 5.0
   - 평가: raw distance
   - GNN 성능에 영향 가능
""")


if __name__ == '__main__':
    main()

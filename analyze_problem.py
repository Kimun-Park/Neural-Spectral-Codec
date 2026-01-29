#!/usr/bin/env python3
"""
Analyze why training validation is misleading
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import yaml
from scipy.spatial.distance import cdist

from data.kitti_loader import KITTILoader
from encoding.spectral_encoder import SpectralEncoder
from keyframe.selector import KeyframeSelector


def main():
    print("=" * 70)
    print("Why Training Validation is Misleading")
    print("=" * 70)

    with open('configs/training_helipr_to_kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)

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
    positions = poses[:, :3, 3]

    # Compute pairwise pose distances
    n = len(keyframes)
    pose_distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=2
    )

    print("\n" + "=" * 70)
    print("Analysis 1: Temporal neighbor distances")
    print("=" * 70)

    # Check distances to temporal neighbors
    temporal_dists = []
    for i in range(n):
        for offset in range(-5, 6):
            if offset == 0:
                continue
            j = i + offset
            if 0 <= j < n:
                temporal_dists.append(pose_distances[i, j])

    temporal_dists = np.array(temporal_dists)
    print(f"\nTemporal neighbors (±5 frames) distance stats:")
    print(f"  Min:    {temporal_dists.min():.2f}m")
    print(f"  Max:    {temporal_dists.max():.2f}m")
    print(f"  Mean:   {temporal_dists.mean():.2f}m")
    print(f"  Median: {np.median(temporal_dists):.2f}m")
    print(f"  % within 5m: {(temporal_dists < 5.0).mean() * 100:.2f}%")

    print("\n" + "=" * 70)
    print("Analysis 2: All-vs-all top-1 is almost always temporal neighbor")
    print("=" * 70)

    # Check what all-vs-all top-1 retrieves (using RAW descriptors, no GNN)
    descriptor_distances = cdist(descriptors, descriptors, metric='euclidean')
    np.fill_diagonal(descriptor_distances, np.inf)

    top1_indices = np.argmin(descriptor_distances, axis=1)
    top1_temporal_offsets = np.abs(top1_indices - np.arange(n))

    print(f"\nDescriptor-only Top-1 retrieval:")
    print(f"  % top-1 is temporal neighbor (offset ≤ 5): {(top1_temporal_offsets <= 5).mean() * 100:.2f}%")
    print(f"  % top-1 is temporal neighbor (offset ≤ 10): {(top1_temporal_offsets <= 10).mean() * 100:.2f}%")
    print(f"  Mean temporal offset: {top1_temporal_offsets.mean():.1f} frames")

    # Check spatial distance to top-1
    top1_spatial_dists = pose_distances[np.arange(n), top1_indices]
    print(f"\n  Top-1 spatial distance stats:")
    print(f"    Min:    {top1_spatial_dists.min():.2f}m")
    print(f"    Max:    {top1_spatial_dists.max():.2f}m")
    print(f"    Mean:   {top1_spatial_dists.mean():.2f}m")
    print(f"    % within 5m: {(top1_spatial_dists < 5.0).mean() * 100:.2f}%")

    print("\n" + "=" * 70)
    print("Analysis 3: Loop closure queries")
    print("=" * 70)

    # Find loop closure queries
    skip_frames = 30
    queries = []
    for i in range(n):
        for j in range(i + skip_frames, n):
            dist = pose_distances[i, j]
            if dist < 5.0:
                queries.append((j, i))
                break

    print(f"\nLoop closure queries (skip_frames={skip_frames}, threshold=5m):")
    print(f"  Number of queries: {len(queries)}")
    print(f"  % of keyframes with revisit: {len(queries) / n * 100:.2f}%")

    if len(queries) > 0:
        # What does descriptor-only retrieve for these queries?
        correct_at_1 = 0
        correct_at_5 = 0
        for query_idx, true_match in queries:
            # Get candidates (excluding temporal neighbors)
            candidates = []
            for i in range(n):
                if abs(i - query_idx) > skip_frames:
                    candidates.append((i, descriptor_distances[query_idx, i], pose_distances[query_idx, i]))

            candidates.sort(key=lambda x: x[1])

            # Check top-1
            if candidates[0][2] < 5.0:
                correct_at_1 += 1

            # Check top-5
            for idx, emb_dist, geo_dist in candidates[:5]:
                if geo_dist < 5.0:
                    correct_at_5 += 1
                    break

        print(f"\n  Descriptor-only R@1: {correct_at_1 / len(queries) * 100:.2f}%")
        print(f"  Descriptor-only R@5: {correct_at_5 / len(queries) * 100:.2f}%")

    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)
    print("""
학습 validation이 99%인 이유:
1. 모든 키프레임을 쿼리로 사용 (all-vs-all)
2. 시간적 인접 프레임 (±5)이 거의 항상 5m 이내
3. Descriptor 유사도가 시간적 인접성과 강하게 상관
4. → Top-1이 거의 항상 시간적 인접 프레임 → 항상 5m 이내

실제 loop closure 성능이 낮은 이유:
1. 시간적 인접 프레임 제외 (skip_frames=30)
2. 실제 "장소 재인식"만 평가
3. Descriptor가 시간적 인접성에 최적화됨 (장소 인식 아님)

근본 문제:
- Triplet loss + all-vs-all validation = 시간적 인접성 학습
- 실제 목표: 장소 재인식 (loop closure)
- 학습 목표와 평가 목표가 일치하지 않음!
""")


if __name__ == '__main__':
    main()

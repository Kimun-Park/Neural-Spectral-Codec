#!/usr/bin/env python3
"""
Analyze trajectory patterns in training vs test data.

Key finding from previous analysis:
- KITTI: 465m x 38m (aspect ratio 12:1, almost 1D)
- HeLiPR: 815m x 853m (aspect ratio ~1:1, 2D)

This might be the root cause of transfer gap.
"""

import sys
sys.path.insert(0, '/workspace/Neural-Spectral-Codec/src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_kitti_poses(sequence):
    """Load KITTI poses"""
    poses_file = Path(f'/workspace/data/kitti/dataset/sequences/{sequence}/poses.txt')
    if not poses_file.exists():
        return None

    poses = []
    with open(poses_file) as f:
        for line in f:
            values = np.array([float(x) for x in line.strip().split()])
            if len(values) == 12:
                pose = np.eye(4)
                pose[:3, :] = values.reshape(3, 4)
                poses.append(pose)
    return poses

def load_helipr_poses(sequence):
    """Load HeLiPR poses"""
    gt_file = Path(f'/workspace/data/helipr/{sequence}/{sequence}/LiDAR_GT/Velodyne_gt.txt')
    if not gt_file.exists():
        return None

    poses = []
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                poses.append(np.array([x, y, z]))
    return [np.array([[1,0,0,p[0]], [0,1,0,p[1]], [0,0,1,p[2]], [0,0,0,1]]) for p in poses]

def analyze_trajectory(poses, name):
    """Analyze trajectory characteristics"""
    if poses is None:
        return None

    positions = np.array([p[:3, 3] for p in poses])

    # Bounding box
    min_xyz = np.min(positions, axis=0)
    max_xyz = np.max(positions, axis=0)
    extent = max_xyz - min_xyz

    # Aspect ratio (XY plane)
    aspect_ratio = max(extent[0], extent[1]) / (min(extent[0], extent[1]) + 1e-6)

    # Total trajectory length
    trajectory_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

    # Area covered
    area = extent[0] * extent[1]

    # Loop closure analysis
    n = len(positions)
    loop_count = 0
    temporal_skip = max(100, n // 100)

    for i in range(0, n, 10):
        for j in range(i + temporal_skip, n, 10):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 5.0:
                loop_count += 1

    # Compactness: how much of the area is actually visited
    grid_size = 5.0  # 5m grid
    visited_cells = set()
    for p in positions:
        gx = int((p[0] - min_xyz[0]) / grid_size)
        gy = int((p[1] - min_xyz[1]) / grid_size)
        visited_cells.add((gx, gy))

    total_cells = int(extent[0] / grid_size + 1) * int(extent[1] / grid_size + 1)
    coverage = len(visited_cells) / total_cells if total_cells > 0 else 0

    print(f"\n{'='*50}")
    print(f"Trajectory Analysis: {name}")
    print(f"{'='*50}")
    print(f"Scans: {n}")
    print(f"Extent: {extent[0]:.1f}m x {extent[1]:.1f}m x {extent[2]:.1f}m")
    print(f"Aspect ratio (XY): {aspect_ratio:.1f}:1")
    print(f"Area: {area:.0f}m²")
    print(f"Trajectory length: {trajectory_length:.0f}m")
    print(f"Coverage: {coverage:.1%} ({len(visited_cells)}/{total_cells} cells)")
    print(f"Loop closures (5m, skip {temporal_skip}): {loop_count}")

    return {
        'name': name,
        'n_scans': n,
        'extent': extent,
        'aspect_ratio': aspect_ratio,
        'area': area,
        'trajectory_length': trajectory_length,
        'coverage': coverage,
        'loop_count': loop_count,
        'positions': positions
    }

def main():
    print("="*60)
    print("Trajectory Pattern Analysis: Training vs Test Data")
    print("="*60)

    # KITTI training sequences
    kitti_train = []
    for seq in ['00', '01', '02', '03', '04', '05', '06', '07', '08']:
        poses = load_kitti_poses(seq)
        if poses:
            result = analyze_trajectory(poses, f'KITTI {seq}')
            if result:
                kitti_train.append(result)

    # KITTI validation
    kitti_val = analyze_trajectory(load_kitti_poses('09'), 'KITTI 09 (val)')

    # HeLiPR test
    helipr_roundabout = analyze_trajectory(load_helipr_poses('Roundabout01'), 'HeLiPR Roundabout01')
    helipr_town = analyze_trajectory(load_helipr_poses('Town01'), 'HeLiPR Town01')

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: Trajectory Characteristics")
    print("="*70)
    print(f"{'Dataset':<25} {'Scans':>8} {'Area':>12} {'Aspect':>8} {'Cover':>8}")
    print("-"*70)

    for r in kitti_train:
        print(f"{r['name']:<25} {r['n_scans']:>8} {r['area']:>10.0f}m² {r['aspect_ratio']:>7.1f}:1 {r['coverage']:>7.1%}")

    if kitti_val:
        print(f"{kitti_val['name']:<25} {kitti_val['n_scans']:>8} {kitti_val['area']:>10.0f}m² {kitti_val['aspect_ratio']:>7.1f}:1 {kitti_val['coverage']:>7.1%}")

    print("-"*70)

    if helipr_roundabout:
        print(f"{helipr_roundabout['name']:<25} {helipr_roundabout['n_scans']:>8} {helipr_roundabout['area']:>10.0f}m² {helipr_roundabout['aspect_ratio']:>7.1f}:1 {helipr_roundabout['coverage']:>7.1%}")
    if helipr_town:
        print(f"{helipr_town['name']:<25} {helipr_town['n_scans']:>8} {helipr_town['area']:>10.0f}m² {helipr_town['aspect_ratio']:>7.1f}:1 {helipr_town['coverage']:>7.1%}")

    # Key insight
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)

    avg_kitti_aspect = np.mean([r['aspect_ratio'] for r in kitti_train])
    avg_kitti_coverage = np.mean([r['coverage'] for r in kitti_train])

    print(f"\nKITTI training average:")
    print(f"  Aspect ratio: {avg_kitti_aspect:.1f}:1")
    print(f"  Coverage: {avg_kitti_coverage:.1%}")

    if helipr_roundabout:
        print(f"\nHeLiPR Roundabout01:")
        print(f"  Aspect ratio: {helipr_roundabout['aspect_ratio']:.1f}:1")
        print(f"  Coverage: {helipr_roundabout['coverage']:.1%}")

        if helipr_roundabout['aspect_ratio'] < avg_kitti_aspect * 0.5:
            print(f"\n→ HeLiPR has {avg_kitti_aspect / helipr_roundabout['aspect_ratio']:.1f}x LOWER aspect ratio")
            print("  KITTI: Narrow corridors (1D-like)")
            print("  HeLiPR: Wide open areas (2D)")
            print("\n→ This means different locations in HeLiPR can be spatially close")
            print("  but approach from completely different directions")

if __name__ == "__main__":
    main()

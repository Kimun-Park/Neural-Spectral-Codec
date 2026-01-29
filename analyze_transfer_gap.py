#!/usr/bin/env python3
"""
Analyze why transfer from KITTI/NCLT to HeLiPR doesn't work well.

Hypothesis to test:
1. KITTI/NCLT have different types of repetitive structures
2. Training data loop closure patterns are different from test
3. FFT pattern distributions differ significantly
"""

import sys
sys.path.insert(0, '/workspace/Neural-Spectral-Codec/src')

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from collections import defaultdict
import os

def load_kitti_data():
    """Load KITTI seq 09 (validation) data"""
    from data.kitti_loader import KITTILoader

    print("Loading KITTI seq 09...")
    loader = KITTILoader('/workspace/data/kitti/dataset', '09', lazy_load=False)
    scans = loader.point_clouds
    poses = loader.poses
    print(f"  Loaded {len(scans)} scans")
    return scans, poses

def load_helipr_roundabout(subsample=None):
    """Load HeLiPR Roundabout01"""
    from data.helipr_loader import HeLiPRLoader

    print("Loading HeLiPR Roundabout01...")
    loader = HeLiPRLoader('/workspace/data/helipr/Roundabout01/Roundabout01', lazy_load=True)

    # Subsample if requested
    if subsample and subsample < len(loader):
        indices = np.linspace(0, len(loader)-1, subsample).astype(int)
    else:
        indices = range(len(loader))

    scans = [loader._load_velodyne(loader.scan_files[i]) for i in indices]
    poses = [loader.scan_poses[i] for i in indices]

    print(f"  Loaded {len(scans)} scans (from {len(loader)} total)")
    return scans, poses

def encode_scans(scans, sensor_type='hdl64e'):
    """Encode scans to spectral histograms"""
    from encoding.spectral_encoder import SpectralEncoder

    if sensor_type == 'hdl64e':
        encoder = SpectralEncoder(
            n_elevation=64, n_azimuth=360, n_bins=50,
            elevation_range=(-24.8, 2.0),
            interpolate_empty=True, learnable_alpha=False
        )
    else:  # vlp16
        encoder = SpectralEncoder(
            n_elevation=16, n_azimuth=360, n_bins=50,
            elevation_range=(-15.0, 15.0),
            interpolate_empty=True, learnable_alpha=False
        )

    histograms = []
    for i, scan in enumerate(scans):
        if i % 500 == 0:
            print(f"  Encoding {i}/{len(scans)}...")
        hist = encoder.encode_points(scan)
        if isinstance(hist, torch.Tensor):
            hist = hist.numpy()
        histograms.append(hist)

    return np.array(histograms)

def compute_distances(poses):
    """Compute pairwise distances from poses"""
    n = len(poses)
    positions = np.array([pose[:3, 3] for pose in poses])

    # Sample pairs for efficiency
    loop_dists = []
    nonloop_dists = []

    np.random.seed(42)

    # Sample 5000 random pairs
    for _ in range(5000):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if abs(i - j) < 100:  # Skip temporal neighbors
            continue

        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < 5.0:  # Loop closure
            loop_dists.append((i, j, dist))
        elif 10.0 < dist < 50.0:  # Non-loop
            nonloop_dists.append((i, j, dist))

    return loop_dists, nonloop_dists

def analyze_discriminability(histograms, loop_pairs, nonloop_pairs, name):
    """Analyze discriminability of histograms"""
    print(f"\n{'='*60}")
    print(f"Discriminability Analysis: {name}")
    print(f"{'='*60}")

    # Compute Wasserstein distances for loop pairs
    loop_wass = []
    for i, j, _ in loop_pairs[:500]:  # Limit for speed
        d = wasserstein_distance(histograms[i], histograms[j])
        loop_wass.append(d)

    # Compute Wasserstein distances for non-loop pairs
    nonloop_wass = []
    for i, j, _ in nonloop_pairs[:500]:
        d = wasserstein_distance(histograms[i], histograms[j])
        nonloop_wass.append(d)

    loop_mean = np.mean(loop_wass)
    loop_std = np.std(loop_wass)
    nonloop_mean = np.mean(nonloop_wass)
    nonloop_std = np.std(nonloop_wass)

    gap = nonloop_mean - loop_mean
    discriminability = gap / loop_std if loop_std > 0 else 0

    print(f"Loop pairs ({len(loop_wass)} samples):")
    print(f"  Mean: {loop_mean:.4f}, Std: {loop_std:.4f}")
    print(f"Non-loop pairs ({len(nonloop_wass)} samples):")
    print(f"  Mean: {nonloop_mean:.4f}, Std: {nonloop_std:.4f}")
    print(f"Gap: {gap:.4f}")
    print(f"Discriminability: {discriminability:.2f}")

    return {
        'loop_mean': loop_mean,
        'loop_std': loop_std,
        'nonloop_mean': nonloop_mean,
        'nonloop_std': nonloop_std,
        'gap': gap,
        'discriminability': discriminability,
        'loop_wass': loop_wass,
        'nonloop_wass': nonloop_wass
    }

def analyze_histogram_distribution(histograms, name):
    """Analyze the distribution of histogram values"""
    print(f"\n{'='*60}")
    print(f"Histogram Distribution: {name}")
    print(f"{'='*60}")

    # Overall statistics
    all_hists = histograms.flatten()
    print(f"Overall: mean={np.mean(all_hists):.4f}, std={np.std(all_hists):.4f}")
    print(f"         min={np.min(all_hists):.4f}, max={np.max(all_hists):.4f}")

    # Per-bin statistics (which bins have most variance?)
    bin_stds = np.std(histograms, axis=0)
    top_var_bins = np.argsort(bin_stds)[-5:][::-1]
    print(f"Top 5 high-variance bins: {top_var_bins}")
    print(f"  Stds: {bin_stds[top_var_bins]}")

    # Histogram similarity within dataset
    # How similar are random pairs?
    n = len(histograms)
    random_pairs_wass = []
    np.random.seed(42)
    for _ in range(1000):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            d = wasserstein_distance(histograms[i], histograms[j])
            random_pairs_wass.append(d)

    print(f"Random pair Wasserstein: mean={np.mean(random_pairs_wass):.4f}, std={np.std(random_pairs_wass):.4f}")

    return {
        'bin_stds': bin_stds,
        'random_pair_mean': np.mean(random_pairs_wass),
        'random_pair_std': np.std(random_pairs_wass)
    }

def analyze_loop_closure_patterns(poses, name):
    """Analyze loop closure spatial patterns"""
    print(f"\n{'='*60}")
    print(f"Loop Closure Patterns: {name}")
    print(f"{'='*60}")

    positions = np.array([pose[:3, 3] for pose in poses])

    # Trajectory length
    total_dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    print(f"Total trajectory length: {total_dist:.1f}m")

    # Area covered (bounding box)
    min_xyz = np.min(positions, axis=0)
    max_xyz = np.max(positions, axis=0)
    area = (max_xyz[0] - min_xyz[0]) * (max_xyz[1] - min_xyz[1])
    print(f"Area covered: {area:.1f}m² ({max_xyz[0]-min_xyz[0]:.1f}m x {max_xyz[1]-min_xyz[1]:.1f}m)")

    # Loop closure density
    n = len(poses)
    loop_count = 0
    skip = 100  # Temporal skip

    for i in range(0, n, 50):  # Sample every 50 frames
        for j in range(i + skip, n, 50):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 5.0:
                loop_count += 1

    print(f"Loop closure count (sampled): {loop_count}")

    # Revisit pattern: how often does the vehicle return to same area?
    grid_size = 10.0  # 10m grid
    grid_visits = defaultdict(list)

    for i, pos in enumerate(positions):
        grid_x = int(pos[0] / grid_size)
        grid_y = int(pos[1] / grid_size)
        grid_visits[(grid_x, grid_y)].append(i)

    revisited_cells = sum(1 for v in grid_visits.values() if len(v) > 1)
    multi_visit_cells = sum(1 for v in grid_visits.values() if len(v) > 2)

    print(f"Grid cells visited: {len(grid_visits)}")
    print(f"Cells revisited (>1 visit): {revisited_cells} ({100*revisited_cells/len(grid_visits):.1f}%)")
    print(f"Cells with 3+ visits: {multi_visit_cells}")

    return {
        'trajectory_length': total_dist,
        'area': area,
        'revisited_ratio': revisited_cells / len(grid_visits) if grid_visits else 0
    }

def main():
    print("="*60)
    print("Transfer Gap Analysis: KITTI vs HeLiPR")
    print("="*60)

    # Load data (subsample HeLiPR to match KITTI size for fair comparison)
    kitti_scans, kitti_poses = load_kitti_data()
    helipr_scans, helipr_poses = load_helipr_roundabout(subsample=1600)  # Similar to KITTI ~1591

    # Encode
    print("\nEncoding KITTI...")
    kitti_hists = encode_scans(kitti_scans, 'hdl64e')

    print("\nEncoding HeLiPR...")
    helipr_hists = encode_scans(helipr_scans, 'vlp16')

    # Already subsampled during load, no need to subsample again
    helipr_hists_sub = helipr_hists
    helipr_poses_sub = helipr_poses

    # Analyze histogram distributions
    kitti_dist = analyze_histogram_distribution(kitti_hists, "KITTI seq 09")
    helipr_dist = analyze_histogram_distribution(helipr_hists_sub, "HeLiPR Roundabout01 (subsampled)")

    # Analyze loop closure patterns
    kitti_patterns = analyze_loop_closure_patterns(kitti_poses, "KITTI seq 09")
    helipr_patterns = analyze_loop_closure_patterns(helipr_poses_sub, "HeLiPR Roundabout01 (subsampled)")

    # Compute discriminability
    kitti_loops, kitti_nonloops = compute_distances(kitti_poses)
    helipr_loops, helipr_nonloops = compute_distances(helipr_poses_sub)

    print(f"\nKITTI: {len(kitti_loops)} loop pairs, {len(kitti_nonloops)} non-loop pairs")
    print(f"HeLiPR: {len(helipr_loops)} loop pairs, {len(helipr_nonloops)} non-loop pairs")

    kitti_disc = analyze_discriminability(kitti_hists, kitti_loops, kitti_nonloops, "KITTI seq 09")
    helipr_disc = analyze_discriminability(helipr_hists_sub, helipr_loops, helipr_nonloops, "HeLiPR Roundabout01")

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Why Transfer Doesn't Work")
    print("="*60)

    print("\n1. HISTOGRAM DISTRIBUTION COMPARISON:")
    print(f"   KITTI random pair mean:  {kitti_dist['random_pair_mean']:.4f}")
    print(f"   HeLiPR random pair mean: {helipr_dist['random_pair_mean']:.4f}")
    print(f"   → HeLiPR histograms are {'more similar' if helipr_dist['random_pair_mean'] < kitti_dist['random_pair_mean'] else 'more different'} to each other")

    print("\n2. LOOP CLOSURE PATTERN COMPARISON:")
    print(f"   KITTI area: {kitti_patterns['area']:.0f}m², revisit ratio: {kitti_patterns['revisited_ratio']:.1%}")
    print(f"   HeLiPR area: {helipr_patterns['area']:.0f}m², revisit ratio: {helipr_patterns['revisited_ratio']:.1%}")

    print("\n3. DISCRIMINABILITY COMPARISON:")
    print(f"   KITTI:  loop={kitti_disc['loop_mean']:.3f}, non-loop={kitti_disc['nonloop_mean']:.3f}, disc={kitti_disc['discriminability']:.2f}")
    print(f"   HeLiPR: loop={helipr_disc['loop_mean']:.3f}, non-loop={helipr_disc['nonloop_mean']:.3f}, disc={helipr_disc['discriminability']:.2f}")

    print("\n4. KEY INSIGHT:")
    if helipr_disc['nonloop_mean'] < kitti_disc['nonloop_mean']:
        print("   → HeLiPR non-loop pairs have SMALLER Wasserstein distance")
        print("   → Different locations in HeLiPR look MORE SIMILAR in FFT space")
        print("   → This is an ENVIRONMENT characteristic, not an algorithm failure")

    # Analyze what makes non-loops similar in HeLiPR
    print("\n5. WHAT MAKES HELIPR LOCATIONS SIMILAR?")

    # Find the most similar non-loop pairs
    similar_nonloops = []
    for i, j, spatial_dist in helipr_nonloops[:200]:
        wass = wasserstein_distance(helipr_hists_sub[i], helipr_hists_sub[j])
        similar_nonloops.append((i, j, spatial_dist, wass))

    similar_nonloops.sort(key=lambda x: x[3])

    print("   Most confusing non-loop pairs (low Wasserstein, high spatial distance):")
    for i, j, spatial_dist, wass in similar_nonloops[:5]:
        print(f"   Frames {i} vs {j}: spatial={spatial_dist:.1f}m, Wass={wass:.4f}")

if __name__ == "__main__":
    main()

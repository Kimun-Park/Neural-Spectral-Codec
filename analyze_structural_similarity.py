#!/usr/bin/env python3
"""
Analyze WHY different locations have similar FFT patterns.

Key question: If FFT is just comparing "same place vs different place",
why does training data matter? What's the fundamental limitation?
"""

import sys
sys.path.insert(0, '/workspace/Neural-Spectral-Codec/src')

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from pathlib import Path

def load_and_encode(dataset_type, path, n_samples=1000):
    """Load data and encode to spectral histograms"""
    from encoding.spectral_encoder import SpectralEncoder

    if dataset_type == 'kitti':
        from data.kitti_loader import KITTILoader
        loader = KITTILoader('/workspace/data/kitti/dataset', path, lazy_load=True)
        n_total = len(loader)
        indices = np.linspace(0, n_total-1, min(n_samples, n_total)).astype(int)

        encoder = SpectralEncoder(
            n_elevation=64, n_azimuth=360, n_bins=50,
            elevation_range=(-24.8, 2.0),
            interpolate_empty=True, learnable_alpha=False
        )

        scans = [loader._load_point_cloud(i) for i in indices]
        poses = [loader.poses[i] for i in indices]

    else:  # helipr
        from data.helipr_loader import HeLiPRLoader
        loader = HeLiPRLoader(path, lazy_load=True)
        n_total = len(loader)
        indices = np.linspace(0, n_total-1, min(n_samples, n_total)).astype(int)

        encoder = SpectralEncoder(
            n_elevation=16, n_azimuth=360, n_bins=50,
            elevation_range=(-15.0, 15.0),
            interpolate_empty=True, learnable_alpha=False
        )

        scans = [loader._load_velodyne(loader.scan_files[i]) for i in indices]
        poses = [loader.scan_poses[i] for i in indices]

    # Encode
    histograms = []
    for scan in scans:
        hist = encoder.encode_points(scan)
        if isinstance(hist, torch.Tensor):
            hist = hist.numpy()
        histograms.append(hist)

    return np.array(histograms), poses

def analyze_confusion(histograms, poses, name, distance_threshold=5.0, skip_frames=50):
    """
    Analyze what causes confusion: when non-loop pairs have low Wasserstein distance
    """
    print(f"\n{'='*70}")
    print(f"Structural Similarity Analysis: {name}")
    print(f"{'='*70}")

    n = len(histograms)
    positions = np.array([p[:3, 3] for p in poses])

    # Collect all pairs with Wasserstein distance
    loop_pairs = []
    nonloop_pairs = []

    for i in range(0, n, 5):  # Sample every 5th frame
        for j in range(i + skip_frames, n, 5):
            spatial_dist = np.linalg.norm(positions[i] - positions[j])
            wass_dist = wasserstein_distance(histograms[i], histograms[j])

            if spatial_dist < distance_threshold:
                loop_pairs.append((i, j, spatial_dist, wass_dist))
            elif spatial_dist > 10.0:  # Clear non-loop
                nonloop_pairs.append((i, j, spatial_dist, wass_dist))

    print(f"Loop pairs: {len(loop_pairs)}, Non-loop pairs: {len(nonloop_pairs)}")

    if len(loop_pairs) == 0 or len(nonloop_pairs) == 0:
        print("Not enough pairs for analysis")
        return None

    # Statistics
    loop_wass = [p[3] for p in loop_pairs]
    nonloop_wass = [p[3] for p in nonloop_pairs]

    loop_mean = np.mean(loop_wass)
    loop_std = np.std(loop_wass)
    nonloop_mean = np.mean(nonloop_wass)
    nonloop_std = np.std(nonloop_wass)

    print(f"\nWasserstein Distance Statistics:")
    print(f"  Loop pairs:     mean={loop_mean:.5f}, std={loop_std:.5f}")
    print(f"  Non-loop pairs: mean={nonloop_mean:.5f}, std={nonloop_std:.5f}")
    print(f"  Gap: {nonloop_mean - loop_mean:.5f}")

    # KEY METRIC: How many non-loop pairs are confused with loop pairs?
    # A non-loop pair is "confused" if its Wasserstein distance is below
    # the threshold that would include 90% of loop pairs
    threshold_90 = np.percentile(loop_wass, 90)
    threshold_95 = np.percentile(loop_wass, 95)

    confused_90 = sum(1 for w in nonloop_wass if w < threshold_90)
    confused_95 = sum(1 for w in nonloop_wass if w < threshold_95)

    confusion_rate_90 = confused_90 / len(nonloop_wass)
    confusion_rate_95 = confused_95 / len(nonloop_wass)

    print(f"\nConfusion Analysis:")
    print(f"  Loop 90th percentile threshold: {threshold_90:.5f}")
    print(f"  Non-loop pairs below this: {confused_90}/{len(nonloop_wass)} ({confusion_rate_90:.1%})")
    print(f"  Loop 95th percentile threshold: {threshold_95:.5f}")
    print(f"  Non-loop pairs below this: {confused_95}/{len(nonloop_wass)} ({confusion_rate_95:.1%})")

    # Find the most confusing non-loop pairs
    nonloop_pairs.sort(key=lambda x: x[3])  # Sort by Wasserstein distance
    print(f"\nMost confusing non-loop pairs (lowest Wasserstein, high spatial distance):")
    for i, j, spatial_dist, wass_dist in nonloop_pairs[:10]:
        print(f"  Frames {i:4d} vs {j:4d}: spatial={spatial_dist:6.1f}m, Wass={wass_dist:.5f}")

    # Distribution overlap
    # What percentage of non-loop distribution overlaps with loop distribution?
    loop_max = np.max(loop_wass)
    nonloop_below_loop_max = sum(1 for w in nonloop_wass if w < loop_max)
    overlap_rate = nonloop_below_loop_max / len(nonloop_wass)

    print(f"\nDistribution Overlap:")
    print(f"  Loop max Wasserstein: {loop_max:.5f}")
    print(f"  Non-loop below loop max: {nonloop_below_loop_max}/{len(nonloop_wass)} ({overlap_rate:.1%})")

    return {
        'loop_mean': loop_mean,
        'loop_std': loop_std,
        'nonloop_mean': nonloop_mean,
        'nonloop_std': nonloop_std,
        'confusion_rate_90': confusion_rate_90,
        'confusion_rate_95': confusion_rate_95,
        'overlap_rate': overlap_rate,
        'threshold_90': threshold_90
    }

def main():
    print("="*70)
    print("Why Different Locations Have Similar FFT Patterns")
    print("="*70)

    results = {}

    # KITTI seq 09 (validation)
    print("\nLoading KITTI 09...")
    kitti_hists, kitti_poses = load_and_encode('kitti', '09', n_samples=1500)
    results['kitti_09'] = analyze_confusion(kitti_hists, kitti_poses, "KITTI 09")

    # HeLiPR Roundabout01
    print("\nLoading HeLiPR Roundabout01...")
    roundabout_hists, roundabout_poses = load_and_encode(
        'helipr', '/workspace/data/helipr/Roundabout01/Roundabout01', n_samples=2000)
    results['roundabout'] = analyze_confusion(roundabout_hists, roundabout_poses, "HeLiPR Roundabout01")

    # HeLiPR Town01
    print("\nLoading HeLiPR Town01...")
    town_hists, town_poses = load_and_encode(
        'helipr', '/workspace/data/helipr/Town01/Town01', n_samples=2000)
    results['town'] = analyze_confusion(town_hists, town_poses, "HeLiPR Town01")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Why FFT-based Place Recognition Fails")
    print("="*70)

    print("\n| Dataset          | Loop Mean | Non-loop Mean | Gap    | Confusion@90% |")
    print("|------------------|-----------|---------------|--------|---------------|")

    for name, r in results.items():
        if r:
            print(f"| {name:<16} | {r['loop_mean']:.5f}   | {r['nonloop_mean']:.5f}       | {r['nonloop_mean']-r['loop_mean']:.5f} | {r['confusion_rate_90']:>12.1%} |")

    print("\nKEY INSIGHT:")
    if results.get('town') and results.get('kitti_09'):
        town_conf = results['town']['confusion_rate_90']
        kitti_conf = results['kitti_09']['confusion_rate_90']

        if town_conf > kitti_conf:
            print(f"  Town01 has {town_conf/kitti_conf:.1f}x higher confusion rate than KITTI")
            print(f"  This means {town_conf:.1%} of different locations in Town01")
            print(f"  have FFT patterns similar to same-location pairs")
            print(f"\n  → FFT fundamentally cannot distinguish locations")
            print(f"     when the ENVIRONMENT STRUCTURE is repetitive")
            print(f"  → This is not about training data, but about the")
            print(f"     descriptor's discriminative power in certain environments")

if __name__ == "__main__":
    main()

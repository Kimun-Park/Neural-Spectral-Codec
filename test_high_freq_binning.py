#!/usr/bin/env python3
"""
Test high-frequency focused binning vs low-frequency focused binning.

Hypothesis: High-frequency bins might capture more detailed environmental
patterns that could help distinguish structurally similar locations.
"""

import sys
sys.path.insert(0, '/workspace/Neural-Spectral-Codec/src')

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from pathlib import Path

class SpectralEncoderCustomAlpha:
    """Spectral encoder with custom alpha for binning"""

    def __init__(self, n_elevation=16, n_azimuth=360, n_bins=50,
                 alpha=2.0, elevation_range=(-15.0, 15.0)):
        self.n_elevation = n_elevation
        self.n_azimuth = n_azimuth
        self.n_bins = n_bins
        self.alpha = alpha
        self.elevation_range = elevation_range
        self.n_freqs = n_azimuth // 2 + 1  # 181

        # Compute bin edges
        self._compute_bin_edges()

    def _compute_bin_edges(self):
        """Compute frequency bin edges based on alpha"""
        t = np.linspace(0, 1, self.n_bins + 1)

        if abs(self.alpha) < 0.01:
            # Linear binning
            self.bin_edges = t * self.n_freqs
        else:
            # Exponential binning
            self.bin_edges = (np.exp(self.alpha * t) - 1) / (np.exp(self.alpha) - 1) * self.n_freqs

    def _project_to_range_image(self, points):
        """Project points to range image"""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Compute spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)  # [-pi, pi]
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))

        # Convert to degrees
        azimuth_deg = np.degrees(azimuth)
        elevation_deg = np.degrees(elevation)

        # Normalize azimuth to [0, 360)
        azimuth_deg = (azimuth_deg + 180) % 360

        # Compute bin indices
        elev_min, elev_max = self.elevation_range
        elev_bins = ((elevation_deg - elev_min) / (elev_max - elev_min) * self.n_elevation).astype(int)
        azim_bins = (azimuth_deg / 360 * self.n_azimuth).astype(int)

        # Clip to valid range
        elev_bins = np.clip(elev_bins, 0, self.n_elevation - 1)
        azim_bins = np.clip(azim_bins, 0, self.n_azimuth - 1)

        # Create range image
        range_image = np.zeros((self.n_elevation, self.n_azimuth))

        # Use minimum range for each bin (closest point)
        for i in range(len(points)):
            eb, ab = elev_bins[i], azim_bins[i]
            if range_image[eb, ab] == 0 or r[i] < range_image[eb, ab]:
                range_image[eb, ab] = r[i]

        # Interpolate empty pixels
        range_image = self._interpolate(range_image)

        return range_image

    def _interpolate(self, range_image):
        """Circular interpolation for empty pixels"""
        for row in range(self.n_elevation):
            row_data = range_image[row]
            valid_mask = row_data > 0

            if not np.any(valid_mask):
                continue
            if np.all(valid_mask):
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_values = row_data[valid_indices]
            invalid_indices = np.where(~valid_mask)[0]

            # Circular extension
            extended_indices = np.concatenate([
                valid_indices - self.n_azimuth,
                valid_indices,
                valid_indices + self.n_azimuth
            ])
            extended_values = np.tile(valid_values, 3)

            # Interpolate
            interpolated = np.interp(invalid_indices, extended_indices, extended_values)
            range_image[row, invalid_indices] = interpolated

        return range_image

    def encode(self, points):
        """Encode point cloud to spectral histogram"""
        # Project to range image
        range_image = self._project_to_range_image(points)

        # FFT along azimuth
        fft_result = np.fft.rfft(range_image, axis=1)
        magnitudes = np.abs(fft_result)  # (n_elevation, n_freqs)

        # Sum across elevation
        magnitude_sum = np.sum(magnitudes, axis=0)  # (n_freqs,)

        # Bin into histogram
        histogram = np.zeros(self.n_bins)
        for freq_idx in range(self.n_freqs):
            bin_idx = np.searchsorted(self.bin_edges, freq_idx) - 1
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            histogram[bin_idx] += magnitude_sum[freq_idx]

        # L2 normalize
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm

        return histogram


def load_helipr_data(path, n_samples=1000):
    """Load HeLiPR data"""
    from data.helipr_loader import HeLiPRLoader

    loader = HeLiPRLoader(path, lazy_load=True)
    n_total = len(loader)
    indices = np.linspace(0, n_total-1, min(n_samples, n_total)).astype(int)

    scans = [loader._load_velodyne(loader.scan_files[i]) for i in indices]
    poses = [loader.scan_poses[i] for i in indices]

    return scans, poses


def evaluate_binning(scans, poses, alpha, name, n_bins=50):
    """Evaluate a specific alpha value"""
    encoder = SpectralEncoderCustomAlpha(
        n_elevation=16, n_azimuth=360, n_bins=n_bins,
        alpha=alpha, elevation_range=(-15.0, 15.0)
    )

    # Encode all scans
    print(f"  Encoding with α={alpha}...")
    histograms = np.array([encoder.encode(scan) for scan in scans])
    positions = np.array([p[:3, 3] for p in poses])

    # Find loop and non-loop pairs
    n = len(histograms)
    loop_pairs = []
    nonloop_pairs = []

    # Use smaller step and skip for subsampled data
    for i in range(0, n, 2):
        for j in range(i + 20, n, 2):  # Reduced skip from 50 to 20
            spatial_dist = np.linalg.norm(positions[i] - positions[j])
            wass_dist = wasserstein_distance(histograms[i], histograms[j])

            if spatial_dist < 5.0:
                loop_pairs.append((i, j, spatial_dist, wass_dist))
            elif spatial_dist > 10.0:
                nonloop_pairs.append((i, j, spatial_dist, wass_dist))

    print(f"  Found {len(loop_pairs)} loop pairs, {len(nonloop_pairs)} non-loop pairs")

    if len(loop_pairs) < 5 or len(nonloop_pairs) < 10:
        return None

    # Compute metrics
    loop_wass = [p[3] for p in loop_pairs]
    nonloop_wass = [p[3] for p in nonloop_pairs]

    loop_mean = np.mean(loop_wass)
    loop_std = np.std(loop_wass)
    nonloop_mean = np.mean(nonloop_wass)

    # Confusion rate at 90th percentile
    threshold_90 = np.percentile(loop_wass, 90)
    confused = sum(1 for w in nonloop_wass if w < threshold_90)
    confusion_rate = confused / len(nonloop_wass)

    # Simulated R@1 (what percentage of loop queries find correct match)
    correct = 0
    total = 0
    for i, j, spatial_dist, loop_wass_dist in loop_pairs[:200]:
        # Find nearest neighbor (excluding temporal neighbors)
        min_dist = float('inf')
        min_idx = -1
        for k in range(n):
            if abs(k - i) < 50:  # Skip temporal neighbors
                continue
            wass = wasserstein_distance(histograms[i], histograms[k])
            if wass < min_dist:
                min_dist = wass
                min_idx = k

        # Check if nearest neighbor is a true positive
        if min_idx >= 0:
            nn_spatial = np.linalg.norm(positions[i] - positions[min_idx])
            if nn_spatial < 5.0:
                correct += 1
        total += 1

    recall_at_1 = correct / total if total > 0 else 0

    print(f"\n{name} (α={alpha:+.1f}):")
    print(f"  Loop pairs: {len(loop_pairs)}, Non-loop pairs: {len(nonloop_pairs)}")
    print(f"  Loop mean: {loop_mean:.5f}, Non-loop mean: {nonloop_mean:.5f}")
    print(f"  Gap: {nonloop_mean - loop_mean:.5f}")
    print(f"  Confusion rate @90%: {confusion_rate:.1%}")
    print(f"  Estimated R@1: {recall_at_1:.1%}")

    return {
        'alpha': alpha,
        'loop_mean': loop_mean,
        'nonloop_mean': nonloop_mean,
        'gap': nonloop_mean - loop_mean,
        'confusion_rate': confusion_rate,
        'recall_at_1': recall_at_1
    }


def visualize_bin_distribution(alphas, n_bins_list=[50, 100]):
    """Show how bins are distributed for different alphas"""
    print("\n" + "="*70)
    print("Bin Distribution Visualization")
    print("="*70)

    n_freqs = 181

    for n_bins in n_bins_list:
        print(f"\n--- {n_bins} bins ---")
        for alpha in alphas:
            t = np.linspace(0, 1, n_bins + 1)
            if abs(alpha) < 0.01:
                edges = t * n_freqs
            else:
                edges = (np.exp(alpha * t) - 1) / (np.exp(alpha) - 1) * n_freqs

            # Count frequencies in low (0-30), mid (30-90), high (90-181)
            low_bins = sum(1 for i in range(n_bins) if edges[i] < 30)
            mid_bins = sum(1 for i in range(n_bins) if 30 <= edges[i] < 90)
            high_bins = sum(1 for i in range(n_bins) if edges[i] >= 90)

            print(f"  α={alpha:+.1f}: Low={low_bins:2d}, Mid={mid_bins:2d}, High={high_bins:2d}")


def main():
    print("="*70)
    print("High-Frequency vs Low-Frequency Binning Comparison")
    print("="*70)

    # Visualize bin distributions
    alphas = [2.0, 0.0, -2.0]
    visualize_bin_distribution(alphas, n_bins_list=[50, 100, 150])

    # Load Town01 data (the hardest case)
    print("\n" + "="*70)
    print("Loading HeLiPR Town01...")
    print("="*70)
    scans, poses = load_helipr_data('/workspace/data/helipr/Town01/Town01', n_samples=2000)
    print(f"Loaded {len(scans)} scans")

    # Test different alpha values and bin counts
    print("\n" + "="*70)
    print("Evaluating Different Binning Strategies")
    print("="*70)

    results = []

    # Test with 50 bins (current)
    print("\n--- 50 bins (current) ---")
    for alpha in [2.0, 0.0]:
        name = "50bins, Low-freq" if alpha == 2.0 else "50bins, Linear"
        r = evaluate_binning(scans, poses, alpha, name, n_bins=50)
        if r:
            r['n_bins'] = 50
            results.append(r)

    # Test with 100 bins
    print("\n--- 100 bins ---")
    for alpha in [2.0, 0.0, -2.0]:
        if alpha == 2.0:
            name = "100bins, Low-freq"
        elif alpha == 0.0:
            name = "100bins, Linear"
        else:
            name = "100bins, High-freq"
        r = evaluate_binning(scans, poses, alpha, name, n_bins=100)
        if r:
            r['n_bins'] = 100
            results.append(r)

    # Test with 150 bins
    print("\n--- 150 bins ---")
    for alpha in [0.0]:
        name = "150bins, Linear"
        r = evaluate_binning(scans, poses, alpha, name, n_bins=150)
        if r:
            r['n_bins'] = 150
            results.append(r)

    # Test with 181 bins (full spectrum, no binning)
    print("\n--- 181 bins (full spectrum) ---")
    for alpha in [0.0]:
        name = "181bins, Full spectrum"
        r = evaluate_binning(scans, poses, alpha, name, n_bins=181)
        if r:
            r['n_bins'] = 181
            results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Bins':>5} {'Strategy':<15} {'α':>6} {'Gap':>10} {'Conf.Rate':>10} {'R@1':>8}")
    print("-"*70)
    for r in results:
        name = "Low-freq" if r['alpha'] > 0 else ("Linear" if r['alpha'] == 0 else "High-freq")
        print(f"{r['n_bins']:>5} {name:<15} {r['alpha']:>+6.1f} {r['gap']:>10.5f} {r['confusion_rate']:>10.1%} {r['recall_at_1']:>8.1%}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    if len(results) >= 2:
        current = next((r for r in results if r['alpha'] == 2.0), None)
        high_freq = next((r for r in results if r['alpha'] == -2.0), None)

        if current and high_freq:
            if high_freq['recall_at_1'] > current['recall_at_1']:
                print(f"✓ High-frequency focus IMPROVES R@1: {current['recall_at_1']:.1%} → {high_freq['recall_at_1']:.1%}")
            else:
                print(f"✗ High-frequency focus does NOT help: {current['recall_at_1']:.1%} → {high_freq['recall_at_1']:.1%}")

            if high_freq['confusion_rate'] < current['confusion_rate']:
                print(f"✓ Confusion rate REDUCED: {current['confusion_rate']:.1%} → {high_freq['confusion_rate']:.1%}")
            else:
                print(f"✗ Confusion rate NOT reduced: {current['confusion_rate']:.1%} → {high_freq['confusion_rate']:.1%}")


if __name__ == "__main__":
    main()

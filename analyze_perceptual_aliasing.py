"""
Perceptual Aliasing Analysis

멀리 떨어진 두 장소에서 descriptor가 비슷하게 나올 때,
실제로 시각적으로도 비슷한지 분석합니다.

출력:
1. False positive pairs (거리 멀지만 유사도 높음)
2. Range image 비교
3. Per-elevation histogram 비교
4. 위치 맵
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.kitti_loader import KITTILoader
from encoding.spectral_encoder import SpectralEncoder
from encoding.range_image import RangeImageProjector, interpolate_range_image


def compute_descriptors(
    loader: KITTILoader,
    encoder: SpectralEncoder,
    max_frames: int = None
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Compute descriptors for all frames

    Returns:
        descriptors: (N, 800) array
        positions: (N, 3) array [x, y, z]
        range_images: list of (H, W) arrays
    """
    descriptors = []
    positions = []
    range_images = []

    n_frames = len(loader) if max_frames is None else min(len(loader), max_frames)

    print(f"Computing descriptors for {n_frames} frames...")

    for i in tqdm(range(n_frames)):
        # Load point cloud and pose
        points = loader.get_point_cloud(i)
        pose = loader.get_pose(i)

        # Extract position from pose
        position = pose[:3, 3]
        positions.append(position)

        # Project to range image
        range_image, _ = encoder.projector.project(points, keep_intensity=False)

        # Interpolate empty pixels
        range_image_interp = interpolate_range_image(range_image, method='linear')
        range_images.append(range_image_interp)

        # Compute descriptor
        range_tensor = torch.from_numpy(range_image_interp).float()
        descriptor = encoder.encode_range_image(range_tensor)
        descriptors.append(descriptor.detach().numpy())

    return np.array(descriptors), np.array(positions), range_images


def find_false_positive_pairs(
    descriptors: np.ndarray,
    positions: np.ndarray,
    min_distance: float = 50.0,
    top_k: int = 20
) -> List[Tuple[int, int, float, float]]:
    """
    Find pairs that are far apart but have similar descriptors

    Returns:
        List of (idx1, idx2, distance, similarity)
    """
    n = len(descriptors)

    # Normalize descriptors for cosine similarity
    desc_norm = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

    print(f"Finding false positive pairs (distance > {min_distance}m)...")

    candidates = []

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            # Compute spatial distance
            dist = np.linalg.norm(positions[i] - positions[j])

            if dist < min_distance:
                continue

            # Compute cosine similarity
            sim = np.dot(desc_norm[i], desc_norm[j])

            candidates.append((i, j, dist, sim))

    # Sort by similarity (descending)
    candidates.sort(key=lambda x: -x[3])

    return candidates[:top_k]


def visualize_pair(
    idx1: int,
    idx2: int,
    range_images: List[np.ndarray],
    descriptors: np.ndarray,
    positions: np.ndarray,
    all_positions: np.ndarray,
    similarity: float,
    distance: float,
    n_elevation: int = 16,
    n_bins: int = 50,
    save_path: str = None
):
    """
    Visualize a pair of frames with their range images and histograms
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])

    # Title
    fig.suptitle(
        f'Perceptual Aliasing Analysis\n'
        f'Frame {idx1} vs Frame {idx2} | Distance: {distance:.1f}m | Similarity: {similarity:.4f}',
        fontsize=14, fontweight='bold'
    )

    # --- Row 1: Range Images ---
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(range_images[idx1], cmap='viridis', aspect='auto')
    ax1.set_title(f'Frame {idx1} - Range Image')
    ax1.set_xlabel('Azimuth')
    ax1.set_ylabel('Elevation')
    plt.colorbar(im1, ax=ax1, label='Range (m)')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(range_images[idx2], cmap='viridis', aspect='auto')
    ax2.set_title(f'Frame {idx2} - Range Image')
    ax2.set_xlabel('Azimuth')
    ax2.set_ylabel('Elevation')
    plt.colorbar(im2, ax=ax2, label='Range (m)')

    # Range image difference
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(range_images[idx1] - range_images[idx2])
    im3 = ax3.imshow(diff, cmap='hot', aspect='auto')
    ax3.set_title('Absolute Difference')
    ax3.set_xlabel('Azimuth')
    ax3.set_ylabel('Elevation')
    plt.colorbar(im3, ax=ax3, label='|Diff| (m)')

    # --- Row 2: Per-Elevation Histograms ---
    hist1 = descriptors[idx1].reshape(n_elevation, n_bins)
    hist2 = descriptors[idx2].reshape(n_elevation, n_bins)

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(hist1, cmap='Blues', aspect='auto')
    ax4.set_title(f'Frame {idx1} - Per-Elevation Histogram')
    ax4.set_xlabel('Frequency Bin')
    ax4.set_ylabel('Elevation')
    plt.colorbar(im4, ax=ax4, label='Energy')

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(hist2, cmap='Blues', aspect='auto')
    ax5.set_title(f'Frame {idx2} - Per-Elevation Histogram')
    ax5.set_xlabel('Frequency Bin')
    ax5.set_ylabel('Elevation')
    plt.colorbar(im5, ax=ax5, label='Energy')

    # Histogram difference
    ax6 = fig.add_subplot(gs[1, 2])
    hist_diff = np.abs(hist1 - hist2)
    im6 = ax6.imshow(hist_diff, cmap='Reds', aspect='auto')
    ax6.set_title(f'Histogram Difference (sum={hist_diff.sum():.4f})')
    ax6.set_xlabel('Frequency Bin')
    ax6.set_ylabel('Elevation')
    plt.colorbar(im6, ax=ax6, label='|Diff|')

    # --- Row 3: Position Map and Elevation Profiles ---
    # Position map
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.scatter(all_positions[:, 0], all_positions[:, 1],
                c='lightgray', s=5, alpha=0.5, label='Trajectory')
    ax7.scatter(positions[idx1, 0], positions[idx1, 1],
                c='red', s=200, marker='*', label=f'Frame {idx1}', zorder=5)
    ax7.scatter(positions[idx2, 0], positions[idx2, 1],
                c='blue', s=200, marker='*', label=f'Frame {idx2}', zorder=5)
    ax7.plot([positions[idx1, 0], positions[idx2, 0]],
             [positions[idx1, 1], positions[idx2, 1]],
             'g--', linewidth=2, label=f'Distance: {distance:.1f}m')
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_title('Position Map (Bird\'s Eye View)')
    ax7.legend(loc='upper right')
    ax7.set_aspect('equal')
    ax7.grid(True, alpha=0.3)

    # Per-elevation energy comparison
    ax8 = fig.add_subplot(gs[2, 2])
    elev_energy1 = hist1.sum(axis=1)
    elev_energy2 = hist2.sum(axis=1)

    x = np.arange(n_elevation)
    width = 0.35
    ax8.bar(x - width/2, elev_energy1, width, label=f'Frame {idx1}', color='red', alpha=0.7)
    ax8.bar(x + width/2, elev_energy2, width, label=f'Frame {idx2}', color='blue', alpha=0.7)
    ax8.set_xlabel('Elevation Index')
    ax8.set_ylabel('Total Energy')
    ax8.set_title('Per-Elevation Energy Distribution')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def analyze_similarity_distribution(
    descriptors: np.ndarray,
    positions: np.ndarray,
    save_path: str = None
):
    """
    Analyze the distribution of similarity vs distance
    """
    n = len(descriptors)

    # Normalize descriptors
    desc_norm = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

    # Sample pairs (full computation too slow for large datasets)
    n_samples = min(100000, n * (n - 1) // 2)

    distances = []
    similarities = []

    print(f"Sampling {n_samples} pairs for distribution analysis...")

    # Random sampling
    np.random.seed(42)
    for _ in tqdm(range(n_samples)):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue

        dist = np.linalg.norm(positions[i] - positions[j])
        sim = np.dot(desc_norm[i], desc_norm[j])

        distances.append(dist)
        similarities.append(sim)

    distances = np.array(distances)
    similarities = np.array(similarities)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Scatter plot
    ax1 = axes[0]
    scatter = ax1.scatter(distances, similarities, c=similarities, cmap='RdYlGn',
                          s=1, alpha=0.3)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Similarity vs Distance')
    ax1.axhline(y=0.9, color='r', linestyle='--', label='High similarity (0.9)')
    ax1.axvline(x=50, color='b', linestyle='--', label='Far distance (50m)')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1)

    # 2. Distance distribution for high similarity pairs
    ax2 = axes[1]
    high_sim_mask = similarities > 0.9
    ax2.hist(distances[high_sim_mask], bins=50, alpha=0.7, color='red',
             label=f'High sim (>{0.9}): {high_sim_mask.sum()} pairs')
    ax2.hist(distances[~high_sim_mask], bins=50, alpha=0.5, color='blue',
             label=f'Low sim: {(~high_sim_mask).sum()} pairs')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distance Distribution by Similarity')
    ax2.legend()

    # 3. Similarity distribution by distance bins
    ax3 = axes[2]
    dist_bins = [0, 10, 30, 50, 100, np.inf]
    colors = plt.cm.viridis(np.linspace(0, 1, len(dist_bins) - 1))

    for i in range(len(dist_bins) - 1):
        mask = (distances >= dist_bins[i]) & (distances < dist_bins[i + 1])
        if mask.sum() > 0:
            label = f'{dist_bins[i]}-{dist_bins[i+1] if dist_bins[i+1] != np.inf else "∞"}m'
            ax3.hist(similarities[mask], bins=50, alpha=0.5, color=colors[i],
                     label=label, density=True)

    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Density')
    ax3.set_title('Similarity Distribution by Distance')
    ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()

    # Print statistics
    print("\n=== Similarity Statistics by Distance ===")
    for i in range(len(dist_bins) - 1):
        mask = (distances >= dist_bins[i]) & (distances < dist_bins[i + 1])
        if mask.sum() > 0:
            sims = similarities[mask]
            print(f"Distance {dist_bins[i]}-{dist_bins[i+1] if dist_bins[i+1] != np.inf else '∞'}m: "
                  f"mean={sims.mean():.4f}, std={sims.std():.4f}, "
                  f"max={sims.max():.4f}, n={mask.sum()}")

    # Count false positives
    fp_mask = (distances > 50) & (similarities > 0.9)
    print(f"\nFalse Positives (dist>50m, sim>0.9): {fp_mask.sum()} / {len(distances)} ({100*fp_mask.sum()/len(distances):.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Perceptual Aliasing Analysis')
    parser.add_argument('--kitti_root', type=str,
                        default='/workspace/data/kitti/dataset',
                        help='KITTI dataset root')
    parser.add_argument('--sequence', type=str, default='00',
                        help='KITTI sequence (00, 09, etc.)')
    parser.add_argument('--max_frames', type=int, default=500,
                        help='Maximum frames to analyze')
    parser.add_argument('--min_distance', type=float, default=50.0,
                        help='Minimum distance for false positive pairs')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top false positive pairs to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/aliasing_analysis',
                        help='Output directory for visualizations')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize encoder
    print("Initializing encoder...")
    encoder = SpectralEncoder(
        n_elevation=64,  # KITTI HDL-64E
        n_azimuth=360,
        n_bins=50,
        target_elevation_bins=16,
        elevation_range=(-24.8, 2.0),
        interpolate_empty=True,
        device='cpu'
    )

    print(f"Encoder output dimension: {encoder.output_dim}")

    # Load KITTI sequence
    print(f"\nLoading KITTI sequence {args.sequence}...")
    loader = KITTILoader(
        data_root=args.kitti_root,
        sequence=args.sequence,
        lazy_load=True
    )
    print(f"Total frames: {len(loader)}")

    # Compute descriptors
    descriptors, positions, range_images = compute_descriptors(
        loader, encoder, max_frames=args.max_frames
    )

    print(f"\nDescriptors shape: {descriptors.shape}")
    print(f"Positions shape: {positions.shape}")

    # Analyze similarity distribution
    print("\n=== Analyzing Similarity Distribution ===")
    analyze_similarity_distribution(
        descriptors, positions,
        save_path=os.path.join(args.output_dir, 'similarity_distribution.png')
    )

    # Find false positive pairs
    print("\n=== Finding False Positive Pairs ===")
    fp_pairs = find_false_positive_pairs(
        descriptors, positions,
        min_distance=args.min_distance,
        top_k=args.top_k
    )

    if len(fp_pairs) == 0:
        print("No false positive pairs found!")
        return

    print(f"\nTop {len(fp_pairs)} false positive pairs:")
    print("-" * 60)
    for rank, (idx1, idx2, dist, sim) in enumerate(fp_pairs):
        print(f"Rank {rank+1}: Frame {idx1} vs {idx2} | "
              f"Distance: {dist:.1f}m | Similarity: {sim:.4f}")

    # Visualize top pairs
    print("\n=== Visualizing Top Pairs ===")
    for rank, (idx1, idx2, dist, sim) in enumerate(fp_pairs):
        save_path = os.path.join(
            args.output_dir,
            f'pair_{rank+1:02d}_f{idx1}_f{idx2}_d{dist:.0f}m_s{sim:.3f}.png'
        )
        visualize_pair(
            idx1, idx2,
            range_images, descriptors, positions, positions,
            sim, dist,
            n_elevation=16,
            n_bins=50,
            save_path=save_path
        )

    # Summary report
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Sequence: KITTI {args.sequence}")
    print(f"Frames analyzed: {len(descriptors)}")
    print(f"False positive pairs found: {len(fp_pairs)}")
    print(f"Output directory: {args.output_dir}")

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Perceptual Aliasing Analysis Summary\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Dataset: KITTI {args.sequence}\n")
        f.write(f"Frames: {len(descriptors)}\n")
        f.write(f"Min distance threshold: {args.min_distance}m\n\n")
        f.write(f"Top False Positive Pairs:\n")
        f.write("-" * 40 + "\n")
        for rank, (idx1, idx2, dist, sim) in enumerate(fp_pairs):
            f.write(f"{rank+1}. Frame {idx1} vs {idx2}: "
                   f"dist={dist:.1f}m, sim={sim:.4f}\n")

    print(f"Summary saved: {summary_path}")


if __name__ == '__main__':
    main()

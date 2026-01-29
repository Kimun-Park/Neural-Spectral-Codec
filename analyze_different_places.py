"""
Different Places Analysis

실제로 다르게 생긴 장소들을 비교하여:
1. Descriptor가 다른 환경을 얼마나 잘 구분하는지
2. Range Image가 시각적으로 얼마나 다른지
확인합니다.

비교 대상:
- KITTI 00의 다양한 구간 (직선 도로, 회전, 건물 근처 등)
- True Negative pairs (거리 멀고 유사도 낮음)
- True Positive pairs (거리 가깝고 유사도 높음) - 기준점
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from tqdm import tqdm
from typing import List, Tuple
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.kitti_loader import KITTILoader
from encoding.spectral_encoder import SpectralEncoder
from encoding.range_image import RangeImageProjector, interpolate_range_image


def compute_descriptors_sparse(
    loader: KITTILoader,
    encoder: SpectralEncoder,
    frame_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Compute descriptors for specific frame indices
    """
    descriptors = []
    positions = []
    range_images = []

    print(f"Computing descriptors for {len(frame_indices)} frames...")

    for i in tqdm(frame_indices):
        points = loader.get_point_cloud(i)
        pose = loader.get_pose(i)

        position = pose[:3, 3]
        positions.append(position)

        range_image, _ = encoder.projector.project(points, keep_intensity=False)
        range_image_interp = interpolate_range_image(range_image, method='linear')
        range_images.append(range_image_interp)

        range_tensor = torch.from_numpy(range_image_interp).float()
        descriptor = encoder.encode_range_image(range_tensor)
        descriptors.append(descriptor.detach().numpy())

    return np.array(descriptors), np.array(positions), range_images


def visualize_comparison(
    frames: List[int],
    range_images: List[np.ndarray],
    descriptors: np.ndarray,
    positions: np.ndarray,
    title: str,
    n_elevation: int = 16,
    n_bins: int = 50,
    save_path: str = None
):
    """
    Compare multiple frames side by side
    """
    n_frames = len(frames)

    fig = plt.figure(figsize=(5 * n_frames, 12))
    gs = GridSpec(4, n_frames, figure=fig, height_ratios=[1, 1, 0.8, 1.2])

    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Compute pairwise similarities
    desc_norm = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)
    sim_matrix = desc_norm @ desc_norm.T

    # Row 1: Range Images
    for i, (frame_idx, ri) in enumerate(zip(frames, range_images)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(ri, cmap='viridis', aspect='auto')
        ax.set_title(f'Frame {frame_idx}', fontsize=11)
        ax.set_xlabel('Azimuth')
        if i == 0:
            ax.set_ylabel('Elevation')
        plt.colorbar(im, ax=ax, label='Range (m)')

    # Row 2: Per-Elevation Histograms
    for i, (frame_idx, desc) in enumerate(zip(frames, descriptors)):
        ax = fig.add_subplot(gs[1, i])
        hist = desc.reshape(n_elevation, n_bins)
        im = ax.imshow(hist, cmap='Blues', aspect='auto')
        ax.set_title(f'Histogram', fontsize=10)
        ax.set_xlabel('Frequency Bin')
        if i == 0:
            ax.set_ylabel('Elevation')
        plt.colorbar(im, ax=ax)

    # Row 3: Similarity matrix
    ax_sim = fig.add_subplot(gs[2, :])
    im = ax_sim.imshow(sim_matrix, cmap='RdYlGn', vmin=0.9, vmax=1.0)
    ax_sim.set_xticks(range(n_frames))
    ax_sim.set_yticks(range(n_frames))
    ax_sim.set_xticklabels([f'F{f}' for f in frames])
    ax_sim.set_yticklabels([f'F{f}' for f in frames])
    ax_sim.set_title('Pairwise Cosine Similarity')

    # Add text annotations
    for i in range(n_frames):
        for j in range(n_frames):
            color = 'white' if sim_matrix[i, j] < 0.95 else 'black'
            ax_sim.text(j, i, f'{sim_matrix[i, j]:.3f}', ha='center', va='center',
                       fontsize=9, color=color)

    plt.colorbar(im, ax=ax_sim)

    # Row 4: Position map
    ax_map = fig.add_subplot(gs[3, :])

    # Plot full trajectory if available
    colors = plt.cm.tab10(np.linspace(0, 1, n_frames))

    for i, (frame_idx, pos) in enumerate(zip(frames, positions)):
        ax_map.scatter(pos[0], pos[1], c=[colors[i]], s=200, marker='*',
                      label=f'Frame {frame_idx}', zorder=5)
        ax_map.annotate(f'F{frame_idx}', (pos[0], pos[1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Draw lines between consecutive frames
    for i in range(n_frames - 1):
        dist = np.linalg.norm(positions[i] - positions[i+1])
        ax_map.plot([positions[i, 0], positions[i+1, 0]],
                   [positions[i, 1], positions[i+1, 1]],
                   '--', color='gray', alpha=0.5)

    ax_map.set_xlabel('X (m)')
    ax_map.set_ylabel('Y (m)')
    ax_map.set_title('Positions (Bird\'s Eye View)')
    ax_map.legend(loc='upper right', fontsize=8)
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def find_diverse_frames(
    loader: KITTILoader,
    encoder: SpectralEncoder,
    n_samples: int = 8
) -> Tuple[List[int], np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Find frames from diverse parts of the trajectory
    """
    total_frames = len(loader)

    # Sample frames evenly across the sequence
    indices = np.linspace(0, total_frames - 1, n_samples, dtype=int).tolist()

    descriptors, positions, range_images = compute_descriptors_sparse(
        loader, encoder, indices
    )

    return indices, descriptors, positions, range_images


def find_extremes(
    loader: KITTILoader,
    encoder: SpectralEncoder,
    n_samples: int = 500
) -> dict:
    """
    Find pairs with extreme similarity values:
    - Most similar (far apart) - False Positives
    - Least similar (far apart) - True Negatives
    - Most similar (close) - True Positives
    """
    total_frames = len(loader)

    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, n_samples, dtype=int).tolist()

    descriptors, positions, range_images = compute_descriptors_sparse(
        loader, encoder, indices
    )

    # Normalize descriptors
    desc_norm = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

    # Find pairs
    n = len(indices)

    false_positives = []  # Far but similar
    true_negatives = []   # Far and different
    true_positives = []   # Close and similar

    print("Finding extreme pairs...")
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            sim = np.dot(desc_norm[i], desc_norm[j])

            if dist > 100:  # Far apart
                if sim > 0.98:
                    false_positives.append((i, j, dist, sim))
                elif sim < 0.95:
                    true_negatives.append((i, j, dist, sim))
            elif dist < 20:  # Close
                if sim > 0.99:
                    true_positives.append((i, j, dist, sim))

    # Sort and get top pairs
    false_positives.sort(key=lambda x: -x[3])  # Highest similarity
    true_negatives.sort(key=lambda x: x[3])     # Lowest similarity
    true_positives.sort(key=lambda x: -x[3])   # Highest similarity

    return {
        'indices': indices,
        'descriptors': descriptors,
        'positions': positions,
        'range_images': range_images,
        'false_positives': false_positives[:5],
        'true_negatives': true_negatives[:5],
        'true_positives': true_positives[:5]
    }


def visualize_pair_comparison(
    idx1: int,
    idx2: int,
    range_images: List[np.ndarray],
    descriptors: np.ndarray,
    positions: np.ndarray,
    indices: List[int],
    similarity: float,
    distance: float,
    pair_type: str,
    n_elevation: int = 16,
    n_bins: int = 50,
    save_path: str = None
):
    """
    Detailed comparison of a single pair
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])

    frame1 = indices[idx1]
    frame2 = indices[idx2]

    # Title with pair type
    type_colors = {
        'False Positive': 'red',
        'True Negative': 'green',
        'True Positive': 'blue'
    }

    fig.suptitle(
        f'{pair_type}: Frame {frame1} vs Frame {frame2}\n'
        f'Distance: {distance:.1f}m | Similarity: {similarity:.4f}',
        fontsize=14, fontweight='bold', color=type_colors.get(pair_type, 'black')
    )

    # Row 1: Range Images
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(range_images[idx1], cmap='viridis', aspect='auto')
    ax1.set_title(f'Frame {frame1}')
    ax1.set_xlabel('Azimuth')
    ax1.set_ylabel('Elevation')
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(range_images[idx2], cmap='viridis', aspect='auto')
    ax2.set_title(f'Frame {frame2}')
    ax2.set_xlabel('Azimuth')
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(range_images[idx1] - range_images[idx2])
    im3 = ax3.imshow(diff, cmap='hot', aspect='auto')
    ax3.set_title(f'Difference (mean={diff.mean():.2f}m)')
    ax3.set_xlabel('Azimuth')
    plt.colorbar(im3, ax=ax3)

    # Row 2: Histograms
    hist1 = descriptors[idx1].reshape(n_elevation, n_bins)
    hist2 = descriptors[idx2].reshape(n_elevation, n_bins)

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(hist1, cmap='Blues', aspect='auto')
    ax4.set_title(f'Histogram {frame1}')
    ax4.set_xlabel('Frequency Bin')
    ax4.set_ylabel('Elevation')
    plt.colorbar(im4, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(hist2, cmap='Blues', aspect='auto')
    ax5.set_title(f'Histogram {frame2}')
    ax5.set_xlabel('Frequency Bin')
    plt.colorbar(im5, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    hist_diff = np.abs(hist1 - hist2)
    im6 = ax6.imshow(hist_diff, cmap='Reds', aspect='auto')
    ax6.set_title(f'Hist Diff (sum={hist_diff.sum():.4f})')
    ax6.set_xlabel('Frequency Bin')
    plt.colorbar(im6, ax=ax6)

    # Row 3: Per-elevation comparison
    ax7 = fig.add_subplot(gs[2, 0:2])
    elev_energy1 = hist1.sum(axis=1)
    elev_energy2 = hist2.sum(axis=1)

    x = np.arange(n_elevation)
    width = 0.35
    ax7.bar(x - width/2, elev_energy1, width, label=f'Frame {frame1}', alpha=0.7)
    ax7.bar(x + width/2, elev_energy2, width, label=f'Frame {frame2}', alpha=0.7)
    ax7.set_xlabel('Elevation Index')
    ax7.set_ylabel('Energy')
    ax7.set_title('Per-Elevation Energy Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Frequency profile comparison
    ax8 = fig.add_subplot(gs[2, 2])
    freq_profile1 = hist1.sum(axis=0)
    freq_profile2 = hist2.sum(axis=0)
    ax8.plot(freq_profile1, label=f'Frame {frame1}', alpha=0.7)
    ax8.plot(freq_profile2, label=f'Frame {frame2}', alpha=0.7)
    ax8.set_xlabel('Frequency Bin')
    ax8.set_ylabel('Energy')
    ax8.set_title('Frequency Profile')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Different Places Analysis')
    parser.add_argument('--kitti_root', type=str,
                        default='/workspace/data/kitti/dataset')
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/different_places_analysis')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize encoder
    print("Initializing encoder...")
    encoder = SpectralEncoder(
        n_elevation=64,
        n_azimuth=360,
        n_bins=50,
        target_elevation_bins=16,
        elevation_range=(-24.8, 2.0),
        interpolate_empty=True,
        device='cpu'
    )

    # Load KITTI
    print(f"\nLoading KITTI sequence {args.sequence}...")
    loader = KITTILoader(
        data_root=args.kitti_root,
        sequence=args.sequence,
        lazy_load=True
    )
    print(f"Total frames: {len(loader)}")

    # === Analysis 1: Diverse frames across trajectory ===
    print("\n" + "="*60)
    print("ANALYSIS 1: Diverse Frames Across Trajectory")
    print("="*60)

    diverse_indices, diverse_desc, diverse_pos, diverse_ri = find_diverse_frames(
        loader, encoder, n_samples=8
    )

    visualize_comparison(
        diverse_indices, diverse_ri, diverse_desc, diverse_pos,
        title=f'KITTI {args.sequence}: Diverse Locations Across Trajectory',
        save_path=os.path.join(args.output_dir, '01_diverse_frames.png')
    )

    # Print similarity stats
    desc_norm = diverse_desc / (np.linalg.norm(diverse_desc, axis=1, keepdims=True) + 1e-8)
    sim_matrix = desc_norm @ desc_norm.T

    print("\nPairwise Similarities:")
    for i in range(len(diverse_indices)):
        for j in range(i+1, len(diverse_indices)):
            dist = np.linalg.norm(diverse_pos[i] - diverse_pos[j])
            print(f"  Frame {diverse_indices[i]:4d} vs {diverse_indices[j]:4d}: "
                  f"sim={sim_matrix[i,j]:.4f}, dist={dist:.1f}m")

    # === Analysis 2: Find extreme pairs ===
    print("\n" + "="*60)
    print("ANALYSIS 2: Extreme Pairs (FP, TN, TP)")
    print("="*60)

    results = find_extremes(loader, encoder, n_samples=300)

    print(f"\nFalse Positives (far but similar): {len(results['false_positives'])}")
    print(f"True Negatives (far and different): {len(results['true_negatives'])}")
    print(f"True Positives (close and similar): {len(results['true_positives'])}")

    # Visualize top pairs from each category
    for pair_type, pairs, color in [
        ('True Negative', results['true_negatives'], 'green'),
        ('False Positive', results['false_positives'], 'red'),
        ('True Positive', results['true_positives'], 'blue')
    ]:
        if len(pairs) == 0:
            print(f"\nNo {pair_type} pairs found")
            continue

        print(f"\n{pair_type} pairs:")
        for rank, (i, j, dist, sim) in enumerate(pairs[:3]):
            frame1 = results['indices'][i]
            frame2 = results['indices'][j]
            print(f"  {rank+1}. Frame {frame1} vs {frame2}: dist={dist:.1f}m, sim={sim:.4f}")

            save_path = os.path.join(
                args.output_dir,
                f'02_{pair_type.lower().replace(" ", "_")}_{rank+1}_f{frame1}_f{frame2}.png'
            )

            visualize_pair_comparison(
                i, j,
                results['range_images'],
                results['descriptors'],
                results['positions'],
                results['indices'],
                sim, dist,
                pair_type,
                save_path=save_path
            )

    # === Analysis 3: Specific interesting locations ===
    print("\n" + "="*60)
    print("ANALYSIS 3: Specific Location Types")
    print("="*60)

    # KITTI 00 specific interesting frames:
    # - 0-200: Starting area (residential)
    # - 1500-1700: Loop closure area
    # - 2500-2700: Highway-like section
    # - 4000-4200: End area

    specific_frames = [50, 500, 1600, 2600, 3500, 4200]
    specific_frames = [f for f in specific_frames if f < len(loader)]

    if len(specific_frames) >= 4:
        spec_desc, spec_pos, spec_ri = compute_descriptors_sparse(
            loader, encoder, specific_frames
        )

        visualize_comparison(
            specific_frames, spec_ri, spec_desc, spec_pos,
            title=f'KITTI {args.sequence}: Specific Location Types',
            save_path=os.path.join(args.output_dir, '03_specific_locations.png')
        )

    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print("\nKey files:")
    print("  01_diverse_frames.png - Evenly sampled across trajectory")
    print("  02_true_negative_*.png - Far and DIFFERENT (good discrimination)")
    print("  02_false_positive_*.png - Far but SIMILAR (aliasing problem)")
    print("  02_true_positive_*.png - Close and similar (expected)")
    print("  03_specific_locations.png - Hand-picked interesting locations")


if __name__ == '__main__':
    main()

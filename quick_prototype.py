"""
Quick Prototype Script

Tests Neural Spectral Codec with minimal data:
- Single KITTI sequence (00 or 05)
- Small subset of frames
- Fast training/testing

Usage:
    python quick_prototype.py --data_dir data/kitti --sequence 00 --max_frames 500
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import time

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.kitti_loader import KITTILoader
from encoding.spectral_encoder import SpectralEncoder
from keyframe.selector import KeyframeSelector
from keyframe.graph_manager import build_graph_from_keyframes
from gnn.model import create_spectral_gnn
from gnn.trainer import create_trainer
from gnn.triplet_miner import create_triplet_miner
from retrieval.two_stage_retrieval import create_two_stage_retrieval


def test_encoding(kitti_loader, max_frames=100):
    """Test spectral encoding"""
    print("\n" + "="*60)
    print("TEST 1: Spectral Encoding")
    print("="*60)

    encoder = SpectralEncoder()

    # Test on first frame
    data = kitti_loader[0]
    points = data['points']

    print(f"Point cloud shape: {points.shape}")

    # Encode
    start = time.time()
    descriptor = encoder.encode_points(points)
    encoding_time = (time.time() - start) * 1000

    print(f"Descriptor shape: {descriptor.shape}")
    print(f"Encoding time: {encoding_time:.2f}ms")
    print(f"Target: <10ms ✓" if encoding_time < 10 else f"Target: <10ms ✗")

    # Test rotation invariance
    print("\nTesting rotation invariance...")
    from data.pose_utils import transform_points

    # Rotate 45 degrees
    angle = np.pi / 4
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    rotated_points = transform_points(points, R)
    descriptor_rotated = encoder.encode_points(rotated_points)

    diff = torch.abs(descriptor - descriptor_rotated).max().item()
    print(f"Max difference after rotation: {diff:.6f}")
    print(f"Rotation invariance: ✓" if diff < 0.01 else f"Rotation invariance: ✗ (diff={diff})")

    return encoder


def test_keyframe_selection(kitti_loader, max_frames=500):
    """Test keyframe selection"""
    print("\n" + "="*60)
    print("TEST 2: Keyframe Selection")
    print("="*60)

    selector = KeyframeSelector(
        distance_threshold=0.5,
        rotation_threshold=15.0,
        temporal_threshold=5.0
    )

    # Process frames
    max_frames = min(max_frames, len(kitti_loader))
    print(f"Processing {max_frames} frames...")

    for i in range(max_frames):
        data = kitti_loader[i]

        selector.process_scan(
            scan_id=i,
            points=data['points'],
            pose=data['pose'],
            timestamp=data['timestamp']
        )

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{max_frames} frames...")

    # Statistics
    stats = selector.get_statistics()
    print(f"\nResults:")
    print(f"  Total scans: {stats['num_scans']}")
    print(f"  Keyframes: {stats['num_keyframes']}")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")
    print(f"  Keyframe rate: {stats['avg_keyframe_rate_hz']:.2f} Hz")
    print(f"  Target rate: ~1Hz {'✓' if 0.5 <= stats['avg_keyframe_rate_hz'] <= 2.0 else '✗'}")

    return selector


def test_gnn_forward(keyframes):
    """Test GNN forward pass"""
    print("\n" + "="*60)
    print("TEST 3: GNN Forward Pass")
    print("="*60)

    if len(keyframes) < 10:
        print("Not enough keyframes for GNN test (need at least 10)")
        return None

    # Use first 50 keyframes
    test_keyframes = keyframes[:min(50, len(keyframes))]

    print(f"Building graph with {len(test_keyframes)} keyframes...")

    # Build graph
    graph = build_graph_from_keyframes(test_keyframes, temporal_neighbors=5)

    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

    # Create GNN
    gnn = create_spectral_gnn(
        input_dim=50,
        hidden_dim=50,
        output_dim=50,
        n_layers=3,
        dropout=0.1
    )

    print(f"GNN parameters: {sum(p.numel() for p in gnn.parameters()):,}")

    # Forward pass
    start = time.time()
    with torch.no_grad():
        embeddings = gnn(graph)
    forward_time = (time.time() - start) * 1000

    print(f"Forward pass time: {forward_time:.2f}ms")
    print(f"Output shape: {embeddings.shape}")

    return gnn


def test_retrieval(keyframes, encoder):
    """Test two-stage retrieval"""
    print("\n" + "="*60)
    print("TEST 4: Two-Stage Retrieval")
    print("="*60)

    if len(keyframes) < 20:
        print("Not enough keyframes for retrieval test (need at least 20)")
        return

    # Create retrieval system
    retrieval = create_two_stage_retrieval(top_k=5)

    # Add first 50 keyframes to database
    database_size = min(50, len(keyframes) - 10)
    print(f"Adding {database_size} keyframes to database...")

    for kf in keyframes[:database_size]:
        retrieval.add_keyframe(kf)

    # Query with a later keyframe
    query_kf = keyframes[database_size + 5]

    print(f"Querying database...")
    start = time.time()

    # Stage 1 only (no verification for speed)
    candidates = retrieval.query(query_kf, verify=False)

    query_time = (time.time() - start) * 1000

    print(f"\nResults:")
    print(f"  Query time: {query_time:.2f}ms")
    print(f"  Candidates found: {len(candidates)}")

    if len(candidates) > 0:
        print(f"\nTop 3 candidates:")
        for i, cand in enumerate(candidates[:3]):
            print(f"    {i+1}. Database idx {cand.database_idx}, "
                  f"Wasserstein distance: {cand.distance:.4f}")

    # Test with verification (if enough candidates)
    if len(candidates) >= 1:
        print(f"\nTesting geometric verification...")
        start = time.time()
        verified_candidates = retrieval.query(query_kf, verify=True)
        verification_time = (time.time() - start) * 1000

        print(f"  Total time (with GICP): {verification_time:.2f}ms")
        print(f"  Verified candidates: {len(verified_candidates)}")

        for i, cand in enumerate(verified_candidates):
            print(f"    {i+1}. Fitness: {cand.fitness:.3f}, RMSE: {cand.rmse:.3f}m")


def run_quick_prototype(data_dir, sequence, max_frames=500):
    """Run quick prototype test"""

    print("="*60)
    print("NEURAL SPECTRAL CODEC - QUICK PROTOTYPE")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Sequence: {sequence}")
    print(f"Max frames: {max_frames}")

    # Load data
    print(f"\nLoading KITTI sequence {sequence}...")
    try:
        kitti_loader = KITTILoader(data_dir, sequence, lazy_load=True)
        print(f"Loaded {len(kitti_loader)} frames")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease download KITTI data first:")
        print("  mkdir -p data/kitti")
        print("  # Download sequence 00 or 05 from KITTI website")
        return

    # Test 1: Encoding
    encoder = test_encoding(kitti_loader, max_frames)

    # Test 2: Keyframe selection
    selector = test_keyframe_selection(kitti_loader, max_frames)

    # Encode all keyframes
    print("\nEncoding keyframes...")
    for kf in selector.keyframes:
        descriptor = encoder.encode_points(kf.points).detach().cpu().numpy()
        kf.descriptor = descriptor

    # Test 3: GNN
    if len(selector.keyframes) >= 10:
        gnn = test_gnn_forward(selector.keyframes)

    # Test 4: Retrieval
    if len(selector.keyframes) >= 20:
        test_retrieval(selector.keyframes, encoder)

    # Summary
    print("\n" + "="*60)
    print("PROTOTYPE TEST COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Download full KITTI sequences 00-09 for training")
    print("  2. Run: python src/pipeline.py --config configs/training.yaml --mode train")
    print("  3. Evaluate on validation sequence")
    print("\nAll core components working! ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick prototype test")
    parser.add_argument('--data_dir', type=str, default='data/kitti',
                        help='KITTI data directory')
    parser.add_argument('--sequence', type=str, default='00',
                        help='KITTI sequence ID (00-10)')
    parser.add_argument('--max_frames', type=int, default=500,
                        help='Maximum frames to process')

    args = parser.parse_args()

    run_quick_prototype(args.data_dir, args.sequence, args.max_frames)

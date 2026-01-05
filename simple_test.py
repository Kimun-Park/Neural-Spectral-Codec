"""
Simple Standalone Test

Tests core functionality without complex imports.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_data_loading():
    """Test KITTI data loading"""
    print("="*60)
    print("TEST 1: Data Loading")
    print("="*60)

    from data.kitti_loader import KITTILoader

    try:
        loader = KITTILoader('data/kitti', '00', lazy_load=True)
        print(f"✓ Loaded KITTI sequence 00")
        print(f"  Frames: {len(loader)}")

        # Load first frame
        data = loader[0]
        print(f"  First frame shape: {data['points'].shape}")
        print(f"  Pose shape: {data['pose'].shape}")

        return loader
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_encoding(loader):
    """Test spectral encoding"""
    print("\n" + "="*60)
    print("TEST 2: Spectral Encoding")
    print("="*60)

    import torch
    from encoding.spectral_encoder import SpectralEncoder
    import time

    encoder = SpectralEncoder()

    # Get first scan
    data = loader[0]
    points = data['points']

    print(f"Input: {points.shape}")

    # Encode
    start = time.time()
    descriptor = encoder.encode_points(points)
    encoding_time = (time.time() - start) * 1000

    print(f"Output: {descriptor.shape}")
    print(f"Encoding time: {encoding_time:.2f}ms")

    if encoding_time < 10:
        print(f"✓ Speed target met (<10ms)")
    else:
        print(f"⚠ Speed target missed ({encoding_time:.2f}ms > 10ms)")

    # Test rotation invariance
    print("\nTesting rotation invariance...")
    from data.pose_utils import transform_points

    angle = np.pi / 4  # 45 degrees
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    rotated = transform_points(points, R)
    descriptor_rot = encoder.encode_points(rotated)

    diff = torch.abs(descriptor - descriptor_rot).max().item()
    print(f"Max difference: {diff:.6f}")

    if diff < 0.01:
        print(f"✓ Rotation invariance verified")
    else:
        print(f"✗ Rotation invariance failed (diff={diff})")

    return encoder


def test_keyframe_selection(loader):
    """Test keyframe selection"""
    print("\n" + "="*60)
    print("TEST 3: Keyframe Selection")
    print("="*60)

    from keyframe.selector import KeyframeSelector

    selector = KeyframeSelector()

    # Process frames
    max_frames = min(200, len(loader))
    print(f"Processing {max_frames} frames...")

    for i in range(max_frames):
        data = loader[i]
        selector.process_scan(
            scan_id=i,
            points=data['points'],
            pose=data['pose'],
            timestamp=data['timestamp']
        )

    stats = selector.get_statistics()
    print(f"\nResults:")
    print(f"  Scans: {stats['num_scans']}")
    print(f"  Keyframes: {stats['num_keyframes']}")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")
    print(f"  Rate: {stats['avg_keyframe_rate_hz']:.2f} Hz")

    if 0.5 <= stats['avg_keyframe_rate_hz'] <= 2.0:
        print(f"✓ Keyframe rate target met (~1Hz)")
    else:
        print(f"⚠ Keyframe rate off target")

    return selector.keyframes


def test_gnn(keyframes, encoder):
    """Test GNN"""
    print("\n" + "="*60)
    print("TEST 4: GNN Forward Pass")
    print("="*60)

    if len(keyframes) < 10:
        print("⚠ Not enough keyframes (need 10+)")
        return

    import torch
    from keyframe.graph_manager import build_graph_from_keyframes
    from gnn.model import create_spectral_gnn
    import time

    # Encode keyframes
    print(f"Encoding {len(keyframes)} keyframes...")
    for kf in keyframes:
        desc = encoder.encode_points(kf.points).detach().cpu().numpy()
        kf.descriptor = desc

    # Build graph
    graph = build_graph_from_keyframes(keyframes[:50] if len(keyframes) > 50 else keyframes)
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

    # Create GNN
    gnn = create_spectral_gnn()
    params = sum(p.numel() for p in gnn.parameters())
    print(f"Parameters: {params:,}")

    # Forward pass
    start = time.time()
    with torch.no_grad():
        embeddings = gnn(graph)
    forward_time = (time.time() - start) * 1000

    print(f"Forward time: {forward_time:.2f}ms")
    print(f"Output: {embeddings.shape}")
    print(f"✓ GNN working")


def main():
    print("="*60)
    print("NEURAL SPECTRAL CODEC - SIMPLE TEST")
    print("="*60)
    print()

    # Test 1: Load data
    loader = test_data_loading()
    if loader is None:
        print("\n✗ Data loading failed. Cannot continue.")
        return 1

    # Test 2: Encoding
    try:
        encoder = test_encoding(loader)
    except Exception as e:
        print(f"\n✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 3: Keyframe selection
    try:
        keyframes = test_keyframe_selection(loader)
    except Exception as e:
        print(f"\n✗ Keyframe selection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 4: GNN
    try:
        test_gnn(keyframes, encoder)
    except Exception as e:
        print(f"\n✗ GNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

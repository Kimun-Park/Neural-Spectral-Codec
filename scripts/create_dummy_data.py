"""
Create Dummy KITTI Data for Testing

Generates synthetic LiDAR scans and poses in KITTI format
without downloading real data. Useful for quick prototyping.
"""

import numpy as np
import argparse
from pathlib import Path


def create_synthetic_lidar_scan(num_points=50000, max_range=80.0):
    """
    Create synthetic LiDAR point cloud

    Simulates HDL-64E LiDAR with realistic-looking data
    """
    # Simulate 64 elevation rings
    n_rings = 64
    points_per_ring = num_points // n_rings

    all_points = []

    for ring_idx in range(n_rings):
        # Elevation angle (from -24.8° to 2.0°)
        elevation = np.deg2rad(-24.8 + ring_idx * (26.8 / n_rings))

        # Azimuth angles (full 360°)
        azimuths = np.linspace(0, 2 * np.pi, points_per_ring)

        # Range (simulate distance with some noise)
        # Create a simple scene: mostly ground plane with some obstacles
        base_range = 10.0 + np.random.randn(points_per_ring) * 2.0

        # Add some "obstacles" at random azimuths
        n_obstacles = 5
        for _ in range(n_obstacles):
            obstacle_azimuth = np.random.uniform(0, 2 * np.pi)
            obstacle_width = np.deg2rad(10)  # 10 degree width

            mask = np.abs(azimuths - obstacle_azimuth) < obstacle_width
            base_range[mask] = np.random.uniform(5, 15)

        # Clip to valid range
        ranges = np.clip(base_range, 1.0, max_range)

        # Convert to Cartesian
        x = ranges * np.cos(elevation) * np.cos(azimuths)
        y = ranges * np.cos(elevation) * np.sin(azimuths)
        z = ranges * np.sin(elevation)

        # Add intensity (random for simplicity)
        intensity = np.random.uniform(0, 1, points_per_ring)

        # Stack
        points = np.stack([x, y, z, intensity], axis=1)
        all_points.append(points)

    # Combine all rings
    all_points = np.vstack(all_points)

    return all_points.astype(np.float32)


def create_trajectory(num_frames, trajectory_type='circle'):
    """
    Create synthetic trajectory

    Args:
        num_frames: Number of poses
        trajectory_type: 'circle', 'straight', or 'loop'

    Returns:
        Array of (num_frames, 4, 4) SE(3) poses
    """
    poses = []

    for i in range(num_frames):
        t = i / num_frames

        if trajectory_type == 'circle':
            # Circular trajectory
            radius = 50.0
            angle = t * 2 * np.pi

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.0

            # Heading tangent to circle
            heading = angle + np.pi / 2

        elif trajectory_type == 'straight':
            # Straight line
            x = t * 100.0
            y = 0.0
            z = 0.0
            heading = 0.0

        elif trajectory_type == 'loop':
            # Figure-8 loop
            x = 50.0 * np.sin(t * 2 * np.pi)
            y = 25.0 * np.sin(t * 4 * np.pi)
            z = 0.0

            # Heading from velocity
            dx = 50.0 * 2 * np.pi * np.cos(t * 2 * np.pi)
            dy = 25.0 * 4 * np.pi * np.cos(t * 4 * np.pi)
            heading = np.arctan2(dy, dx)

        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        # Create SE(3) pose
        pose = np.eye(4)

        # Rotation (yaw only)
        c, s = np.cos(heading), np.sin(heading)
        pose[:3, :3] = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

        # Translation
        pose[:3, 3] = [x, y, z]

        poses.append(pose)

    return np.array(poses)


def save_kitti_format(output_dir, num_frames=1000, trajectory_type='circle'):
    """
    Save data in KITTI format

    Args:
        output_dir: Output directory
        num_frames: Number of frames to generate
        trajectory_type: Trajectory type
    """
    output_dir = Path(output_dir)
    sequence_dir = output_dir / 'sequences' / '00'
    velodyne_dir = sequence_dir / 'velodyne'

    # Create directories
    velodyne_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_frames} synthetic LiDAR scans...")

    # Generate trajectory
    poses = create_trajectory(num_frames, trajectory_type)

    # Save each scan
    for i in range(num_frames):
        # Generate point cloud
        points = create_synthetic_lidar_scan()

        # Save as .bin file
        bin_file = velodyne_dir / f"{i:06d}.bin"
        points.tofile(bin_file)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_frames} scans...")

    # Save poses
    poses_file = sequence_dir / 'poses.txt'
    with open(poses_file, 'w') as f:
        for pose in poses:
            # KITTI format: 3x4 transformation matrix as single line
            pose_3x4 = pose[:3, :].flatten()
            line = ' '.join([f"{x:.6f}" for x in pose_3x4])
            f.write(line + '\n')

    print(f"\nDummy data created successfully!")
    print(f"  Location: {sequence_dir}")
    print(f"  Frames: {num_frames}")
    print(f"  Size: ~{num_frames * 0.8:.1f} MB")
    print(f"\nYou can now run:")
    print(f"  python quick_prototype.py --data_dir {output_dir} --sequence 00")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dummy KITTI data")
    parser.add_argument('--output', type=str, default='data/kitti',
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='Number of frames to generate')
    parser.add_argument('--trajectory', type=str, default='circle',
                        choices=['circle', 'straight', 'loop'],
                        help='Trajectory type')

    args = parser.parse_args()

    save_kitti_format(args.output, args.num_frames, args.trajectory)

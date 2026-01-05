"""
Check Setup Script

Verifies that all dependencies and data are ready for running the prototype.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependencies():
    """Check required packages"""
    print("\nChecking dependencies...")

    required = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'open3d': 'Open3D',
        'yaml': 'PyYAML',
        'h5py': 'h5py'
    }

    all_ok = True

    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            all_ok = False

    return all_ok


def check_data():
    """Check if KITTI data exists"""
    print("\nChecking KITTI data...")

    data_dir = Path('data/kitti')

    if not data_dir.exists():
        print(f"  ✗ Data directory not found: {data_dir}")
        print(f"\n  Options:")
        print(f"    1. Create dummy data: python scripts/create_dummy_data.py")
        print(f"    2. Download KITTI: see QUICKSTART.md")
        return False

    # Check for sequence 00
    seq_dir = data_dir / 'sequences' / '00'

    if not seq_dir.exists():
        print(f"  ✗ Sequence 00 not found: {seq_dir}")
        return False

    velodyne_dir = seq_dir / 'velodyne'
    poses_file = seq_dir / 'poses.txt'

    if not velodyne_dir.exists():
        print(f"  ✗ Velodyne directory not found: {velodyne_dir}")
        return False

    if not poses_file.exists():
        print(f"  ✗ Poses file not found: {poses_file}")
        return False

    # Count scans
    scans = list(velodyne_dir.glob('*.bin'))
    print(f"  ✓ Found {len(scans)} scans in sequence 00")

    return True


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU available: {device_name}")
            print(f"    CUDA version: {torch.version.cuda}")
            return True
        else:
            print(f"  ⚠ No GPU available (will use CPU)")
            return False
    except ImportError:
        print(f"  ⚠ PyTorch not installed (can't check GPU)")
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("NEURAL SPECTRAL CODEC - SETUP CHECK")
    print("="*60)

    results = {
        'Python version': check_python_version(),
        'Dependencies': check_dependencies(),
        'KITTI data': check_data(),
        'GPU': check_gpu()
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, ok in results.items():
        status = "✓" if ok else ("✗" if name != "GPU" else "⚠")
        print(f"{status} {name}")

    # Overall status
    critical_ok = results['Python version'] and results['Dependencies']

    if not critical_ok:
        print("\n❌ Setup incomplete. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return 1

    if not results['KITTI data']:
        print("\n⚠ No KITTI data found. You can:")
        print("   1. Create dummy data: python scripts/create_dummy_data.py")
        print("   2. Download KITTI: see QUICKSTART.md")
        return 1

    print("\n✅ Setup complete! You can now run:")
    print("   python quick_prototype.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Robust HeLiPR Dataset Downloader

Handles Google Drive rate limiting with:
- Exponential backoff on failures
- Long delays between downloads
- Resume capability
- Only downloads essential files (Velodyne.tar.gz + Velodyne_gt.txt)
"""

import os
import subprocess
import time
import tarfile
from pathlib import Path

# Sequence folder IDs from Google Drive
SEQUENCES = {
    # Already available
    # 'Town01': '...',
    # 'Roundabout01': '...',

    # To download
    'Roundabout02': '1zDfXFh8A5p0qSI8m_zIqD_pYHiUtzIBf',
    'Roundabout03': '1zgbQM5K5M4Ge2qfuD72a7_qwcFDINSSD',
    'Town02': '1_4QG1pbemtpc_9AYfsqfFzuOXyVG8AvD',
    'Town03': '1Cp3iyo1cIkE3haeVaZ1S2CL9GKa9gkNw',
    'Bridge01': '1805PvGgtXyVMA4IvVwFeLjtVCI2sEkAD',
    'Bridge02': '1BG1VIaZyU6EbvVi-PKhlOgxE5pjHAKd2',
    'Bridge03': '10wcI3NnmpCiyfJWSUGP8Z4-izT7qRFjH',
    'Bridge04': '1fLfJv8BKzqkt3OwtqeGZkfTvgU0F3s2M',
    'KAIST04': '1uTNh1ruKc58PcwS9i_zDjTG1yDkfmSjk',
    'KAIST05': '1akiyFX5XypE5Nu9a97Zc3ADnhHsQcvfY',
    'KAIST06': '1G8JjFThNxTXqGJGa0IrSabiiluk0gAZ0',
    'DCC04': '1nSOGrn0deQp66jrMcbyfMA-a4hqztgD9',
    'DCC05': '18_rsGMYQwWrMeL7Nt9pRR-4pDIr5k0Tc',
    'DCC06': '1siJ4v56sM7njspm0STOA6vrlROtEKmjY',
    'Riverside04': '1DTNJt_uSivnPJY2-kgB0mo4FnKjm78Hh',
    'Riverside05': '1XbAjJfZkug9ZIUADBmvvcxG-ymVc887N',
    'Riverside06': '1rU3c17ny-ZRbSq4Px4qt4l0BKnR5WmJp',
}

BASE_DIR = Path('/workspace/data/helipr')
DELAY_BETWEEN_SEQUENCES = 120  # 2 minutes between sequences
DELAY_ON_RATE_LIMIT = 300      # 5 minutes on rate limit error
MAX_RETRIES = 5


def is_sequence_complete(seq_name):
    """Check if sequence is already downloaded and extracted"""
    seq_dir = BASE_DIR / seq_name / seq_name
    velodyne_dir = seq_dir / 'LiDAR' / 'Velodyne'
    gt_file = seq_dir / 'LiDAR_GT' / 'Velodyne_gt.txt'

    if not velodyne_dir.exists():
        return False

    # Check if there are enough files (at least 1000 scans expected)
    scan_count = len(list(velodyne_dir.glob('*.bin')))
    if scan_count < 1000:
        return False

    if not gt_file.exists():
        return False

    return True


def download_folder(folder_id, output_dir, seq_name):
    """Download folder using gdown with retry logic"""
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    for attempt in range(MAX_RETRIES):
        try:
            print(f"\n  Attempt {attempt + 1}/{MAX_RETRIES}")

            # Use gdown to download folder
            cmd = [
                'gdown', '--folder', url,
                '-O', str(output_dir),
                '--remaining-ok'  # Continue even if some files fail
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Check for rate limit errors
            if 'Too many users' in result.stderr or 'quota' in result.stderr.lower():
                print(f"  Rate limited! Waiting {DELAY_ON_RATE_LIMIT}s...")
                time.sleep(DELAY_ON_RATE_LIMIT)
                continue

            # Check for success
            if result.returncode == 0:
                print(f"  Download successful!")
                return True
            else:
                print(f"  Download failed: {result.stderr[:200]}")
                time.sleep(60)  # Wait 1 minute before retry

        except subprocess.TimeoutExpired:
            print(f"  Timeout! Retrying...")
            time.sleep(60)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(60)

    return False


def extract_velodyne(seq_dir):
    """Extract Velodyne.tar.gz if it exists"""
    tar_path = seq_dir / 'LiDAR' / 'Velodyne.tar.gz'

    if not tar_path.exists():
        print(f"  No tar.gz file found")
        return False

    print(f"  Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=seq_dir / 'LiDAR')

        # Remove tar.gz after successful extraction
        tar_path.unlink()
        print(f"  Extraction complete!")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


def download_sequence(seq_name, folder_id):
    """Download and extract a single sequence"""
    print(f"\n{'='*60}")
    print(f"Processing: {seq_name}")
    print(f"{'='*60}")

    # Check if already complete
    if is_sequence_complete(seq_name):
        print(f"  Already complete! Skipping...")
        return True

    seq_dir = BASE_DIR / seq_name / seq_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Download folder
    print(f"  Downloading from Google Drive...")
    success = download_folder(folder_id, seq_dir, seq_name)

    if not success:
        print(f"  WARNING: Download may be incomplete!")

    # Extract Velodyne.tar.gz
    extract_velodyne(seq_dir)

    # Verify
    velodyne_dir = seq_dir / 'LiDAR' / 'Velodyne'
    gt_file = seq_dir / 'LiDAR_GT' / 'Velodyne_gt.txt'

    if velodyne_dir.exists():
        scan_count = len(list(velodyne_dir.glob('*.bin')))
        gt_status = "Yes" if gt_file.exists() else "No"
        print(f"  Result: {scan_count} scans, GT={gt_status}")
        return scan_count > 0
    else:
        print(f"  Result: No Velodyne data found")
        return False


def main():
    print("="*60)
    print("HeLiPR Dataset Robust Downloader")
    print("="*60)
    print(f"Target directory: {BASE_DIR}")
    print(f"Sequences to download: {len(SEQUENCES)}")
    print(f"Delay between sequences: {DELAY_BETWEEN_SEQUENCES}s")
    print()

    # Check existing data
    print("Existing data check:")
    for seq_name in ['Town01', 'Roundabout01'] + list(SEQUENCES.keys()):
        seq_dir = BASE_DIR / seq_name / seq_name / 'LiDAR' / 'Velodyne'
        if seq_dir.exists():
            count = len(list(seq_dir.glob('*.bin')))
            print(f"  {seq_name}: {count} scans")
        else:
            print(f"  {seq_name}: Not downloaded")
    print()

    # Download sequences
    results = {}
    for seq_name, folder_id in SEQUENCES.items():
        success = download_sequence(seq_name, folder_id)
        results[seq_name] = success

        if success:
            print(f"  Waiting {DELAY_BETWEEN_SEQUENCES}s before next sequence...")
            time.sleep(DELAY_BETWEEN_SEQUENCES)
        else:
            print(f"  Waiting {DELAY_ON_RATE_LIMIT}s due to failure...")
            time.sleep(DELAY_ON_RATE_LIMIT)

    # Final summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    for seq_name in ['Town01', 'Roundabout01'] + list(SEQUENCES.keys()):
        seq_dir = BASE_DIR / seq_name / seq_name
        velodyne_dir = seq_dir / 'LiDAR' / 'Velodyne'
        gt_file = seq_dir / 'LiDAR_GT' / 'Velodyne_gt.txt'

        if velodyne_dir.exists():
            count = len(list(velodyne_dir.glob('*.bin')))
            gt = "Yes" if gt_file.exists() else "No"
            status = "OK" if count > 1000 else "PARTIAL"
            print(f"  {seq_name}: {count} scans, GT={gt} [{status}]")
        else:
            print(f"  {seq_name}: NOT DOWNLOADED")


if __name__ == '__main__':
    main()

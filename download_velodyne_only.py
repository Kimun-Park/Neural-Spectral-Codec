#!/usr/bin/env python3
"""
Download only Velodyne files from HeLiPR dataset.
Uses specific file IDs and long delays to avoid rate limiting.
"""

import os
import subprocess
import time
import tarfile
from pathlib import Path
import sys

# Velodyne file IDs extracted from folder structure
# Format: seq_name: (velodyne_tar_id, velodyne_gt_id)
VELODYNE_FILES = {
    # Town sequences
    'Town02': ('1a12Zu7bMaQfkEwHMR6NYWOpgPUgli96p', '1dPnITl-jwLHfplze6uqx-t4S27andnqi'),
    'Town03': (None, None),  # Will get from folder

    # Roundabout sequences
    'Roundabout02': (None, None),
    'Roundabout03': (None, None),

    # Bridge sequences
    'Bridge01': ('1O--xJnnvZyBjGZGGbItPxkOs-em4LfWU', '1zyJQS8xCHRD-_dcbLyn-d5IigS8U9ldC'),
    'Bridge02': (None, None),
    'Bridge03': (None, None),
    'Bridge04': (None, None),

    # KAIST sequences
    'KAIST04': (None, None),
    'KAIST05': (None, None),
    'KAIST06': (None, None),

    # DCC sequences
    'DCC04': (None, None),
    'DCC05': (None, None),
    'DCC06': (None, None),

    # Riverside sequences
    'Riverside04': (None, None),
    'Riverside05': (None, None),
    'Riverside06': (None, None),
}

# Folder IDs for getting file IDs if not known
FOLDER_IDS = {
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

# Delays (in seconds)
DELAY_AFTER_DOWNLOAD = 120   # 2 minutes after each file
DELAY_ON_RATE_LIMIT = 600    # 10 minutes on rate limit
DELAY_BETWEEN_SEQUENCES = 300  # 5 minutes between sequences


def download_file(file_id, output_path, max_retries=5):
    """Download a single file from Google Drive with retry logic"""
    url = f"https://drive.google.com/uc?id={file_id}"

    for attempt in range(max_retries):
        print(f"    Attempt {attempt + 1}/{max_retries}: {output_path.name}")

        try:
            result = subprocess.run(
                ['gdown', url, '-O', str(output_path), '--fuzzy'],
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )

            # Check for rate limit
            if 'Too many users' in result.stderr or 'quota' in result.stderr.lower():
                print(f"    Rate limited! Waiting {DELAY_ON_RATE_LIMIT}s...")
                time.sleep(DELAY_ON_RATE_LIMIT)
                continue

            # Check for success
            if output_path.exists() and output_path.stat().st_size > 1000:
                print(f"    Success! Size: {output_path.stat().st_size / 1e6:.1f} MB")
                return True
            else:
                print(f"    Failed or incomplete")
                time.sleep(60)

        except subprocess.TimeoutExpired:
            print(f"    Timeout!")
            time.sleep(60)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(60)

    return False


def get_file_ids_from_folder(folder_id, seq_name):
    """Get Velodyne file IDs from folder listing"""
    print(f"  Getting file IDs from folder...")

    # Use gdown to list folder contents
    import tempfile
    import json

    try:
        # Download folder structure only (dry-run style)
        result = subprocess.run(
            ['gdown', '--folder', f'https://drive.google.com/drive/folders/{folder_id}',
             '--dry-run', '-O', '/dev/null'],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Parse output for file IDs
        output = result.stdout + result.stderr

        velodyne_tar_id = None
        velodyne_gt_id = None

        for line in output.split('\n'):
            if 'Velodyne.tar.gz' in line and 'Processing file' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'file':
                        velodyne_tar_id = parts[i + 1]
                        break
            if 'Velodyne_gt.txt' in line and 'Processing file' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'file':
                        velodyne_gt_id = parts[i + 1]
                        break

        return velodyne_tar_id, velodyne_gt_id

    except Exception as e:
        print(f"  Error getting file IDs: {e}")
        return None, None


def is_complete(seq_name):
    """Check if sequence is already complete"""
    seq_dir = BASE_DIR / seq_name / seq_name
    velodyne_dir = seq_dir / 'LiDAR' / 'Velodyne'
    gt_file = seq_dir / 'LiDAR_GT' / 'Velodyne_gt.txt'

    if not velodyne_dir.exists():
        return False

    scan_count = len(list(velodyne_dir.glob('*.bin')))
    if scan_count < 1000:
        return False

    if not gt_file.exists():
        return False

    return True


def download_sequence(seq_name):
    """Download Velodyne files for a sequence"""
    print(f"\n{'='*60}")
    print(f"Sequence: {seq_name}")
    print(f"{'='*60}")

    if is_complete(seq_name):
        print(f"  Already complete! Skipping...")
        return True

    seq_dir = BASE_DIR / seq_name / seq_name
    lidar_dir = seq_dir / 'LiDAR'
    gt_dir = seq_dir / 'LiDAR_GT'
    lidar_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Get file IDs
    tar_id, gt_id = VELODYNE_FILES.get(seq_name, (None, None))

    if tar_id is None or gt_id is None:
        folder_id = FOLDER_IDS.get(seq_name)
        if folder_id:
            tar_id_new, gt_id_new = get_file_ids_from_folder(folder_id, seq_name)
            if tar_id is None:
                tar_id = tar_id_new
            if gt_id is None:
                gt_id = gt_id_new

    if tar_id is None:
        print(f"  ERROR: Could not find Velodyne.tar.gz file ID")
        return False

    # Download Velodyne.tar.gz
    tar_path = lidar_dir / 'Velodyne.tar.gz'
    velodyne_dir = lidar_dir / 'Velodyne'

    if not velodyne_dir.exists() or len(list(velodyne_dir.glob('*.bin'))) < 100:
        if not tar_path.exists():
            print(f"  Downloading Velodyne.tar.gz...")
            success = download_file(tar_id, tar_path)
            if success:
                time.sleep(DELAY_AFTER_DOWNLOAD)
            else:
                print(f"  Failed to download Velodyne.tar.gz")
                return False

        # Extract
        if tar_path.exists():
            print(f"  Extracting tar.gz...")
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=lidar_dir)
                tar_path.unlink()
                print(f"  Extraction complete!")
            except Exception as e:
                print(f"  Extraction error: {e}")

    # Download Velodyne_gt.txt
    gt_path = gt_dir / 'Velodyne_gt.txt'
    if not gt_path.exists() and gt_id:
        print(f"  Downloading Velodyne_gt.txt...")
        success = download_file(gt_id, gt_path)
        if success:
            time.sleep(DELAY_AFTER_DOWNLOAD)

    # Verify
    if velodyne_dir.exists():
        scan_count = len(list(velodyne_dir.glob('*.bin')))
        gt_exists = gt_path.exists()
        print(f"  Result: {scan_count} scans, GT={'Yes' if gt_exists else 'No'}")
        return scan_count > 1000 and gt_exists

    return False


def main():
    print("="*60)
    print("HeLiPR Velodyne-Only Downloader")
    print("="*60)
    print(f"Delay after each file: {DELAY_AFTER_DOWNLOAD}s")
    print(f"Delay on rate limit: {DELAY_ON_RATE_LIMIT}s")
    print(f"Delay between sequences: {DELAY_BETWEEN_SEQUENCES}s")
    print()

    # Status check
    print("Current status:")
    for seq in ['Town01', 'Town02', 'Town03', 'Roundabout01', 'Roundabout02', 'Roundabout03',
                'Bridge01', 'Bridge02', 'Bridge03', 'Bridge04',
                'KAIST04', 'KAIST05', 'KAIST06',
                'DCC04', 'DCC05', 'DCC06',
                'Riverside04', 'Riverside05', 'Riverside06']:
        if is_complete(seq):
            print(f"  {seq}: COMPLETE")
        else:
            print(f"  {seq}: PENDING")
    print()

    # Download pending sequences
    sequences_to_download = list(FOLDER_IDS.keys())

    for seq_name in sequences_to_download:
        success = download_sequence(seq_name)

        if success:
            print(f"  Waiting {DELAY_BETWEEN_SEQUENCES}s before next sequence...")
            time.sleep(DELAY_BETWEEN_SEQUENCES)
        else:
            print(f"  Failed! Waiting {DELAY_ON_RATE_LIMIT}s...")
            time.sleep(DELAY_ON_RATE_LIMIT)

    # Final summary
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)

    complete = 0
    for seq in ['Town01', 'Town02', 'Town03', 'Roundabout01', 'Roundabout02', 'Roundabout03',
                'Bridge01', 'Bridge02', 'Bridge03', 'Bridge04',
                'KAIST04', 'KAIST05', 'KAIST06',
                'DCC04', 'DCC05', 'DCC06',
                'Riverside04', 'Riverside05', 'Riverside06']:
        seq_dir = BASE_DIR / seq / seq
        velodyne_dir = seq_dir / 'LiDAR' / 'Velodyne'
        gt_file = seq_dir / 'LiDAR_GT' / 'Velodyne_gt.txt'

        if velodyne_dir.exists():
            count = len(list(velodyne_dir.glob('*.bin')))
            gt = "Yes" if gt_file.exists() else "No"
            status = "OK" if count > 1000 else "PARTIAL"
            print(f"  {seq}: {count} scans, GT={gt} [{status}]")
            if count > 1000:
                complete += 1
        else:
            print(f"  {seq}: NOT DOWNLOADED")

    print(f"\nComplete: {complete}/19 sequences")


if __name__ == '__main__':
    main()

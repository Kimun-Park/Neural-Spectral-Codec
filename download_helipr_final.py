#!/usr/bin/env python3
"""
Final HeLiPR Velodyne Downloader

1. Lists folder contents to get file IDs
2. Downloads only Velodyne.tar.gz and Velodyne_gt.txt
3. Extracts and cleans up
"""

import os
import subprocess
import time
import tarfile
import re
from pathlib import Path

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

DELAY_AFTER_FILE = 180       # 3 minutes after each file
DELAY_ON_RATE_LIMIT = 900    # 15 minutes on rate limit
DELAY_BETWEEN_SEQS = 300     # 5 minutes between sequences
MAX_RETRIES = 5


def get_velodyne_ids(folder_id):
    """Get Velodyne file IDs by listing folder"""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        proc = subprocess.Popen(
            ['gdown', '--folder',
             f'https://drive.google.com/drive/folders/{folder_id}',
             '-O', temp_dir, '--remaining-ok'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        velodyne_tar_id = None
        velodyne_gt_id = None

        # Read output line by line
        start_time = time.time()
        while True:
            line = proc.stdout.readline()
            if not line:
                break

            # Parse file IDs
            if 'Processing file' in line and 'Velodyne.tar.gz' in line:
                match = re.search(r'Processing file (\S+) Velodyne\.tar\.gz', line)
                if match:
                    velodyne_tar_id = match.group(1)
                    print(f"    Found Velodyne.tar.gz: {velodyne_tar_id}")

            if 'Processing file' in line and 'Velodyne_gt.txt' in line:
                match = re.search(r'Processing file (\S+) Velodyne_gt\.txt', line)
                if match:
                    velodyne_gt_id = match.group(1)
                    print(f"    Found Velodyne_gt.txt: {velodyne_gt_id}")

            # If we found both, kill the process
            if velodyne_tar_id and velodyne_gt_id:
                proc.terminate()
                break

            # Timeout after 60 seconds
            if time.time() - start_time > 60:
                proc.terminate()
                break

        proc.wait()
        return velodyne_tar_id, velodyne_gt_id

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def download_file(file_id, output_path):
    """Download a single file with retry"""
    url = f"https://drive.google.com/uc?id={file_id}"

    for attempt in range(MAX_RETRIES):
        print(f"    Download attempt {attempt + 1}/{MAX_RETRIES}")

        try:
            result = subprocess.run(
                ['gdown', url, '-O', str(output_path), '--fuzzy'],
                capture_output=True,
                text=True,
                timeout=3600
            )

            output = result.stdout + result.stderr

            # Check rate limit
            if 'Too many users' in output or 'quota' in output.lower():
                print(f"    Rate limited! Waiting {DELAY_ON_RATE_LIMIT}s...")
                time.sleep(DELAY_ON_RATE_LIMIT)
                continue

            # Check success
            if output_path.exists() and output_path.stat().st_size > 1000:
                size_mb = output_path.stat().st_size / 1e6
                print(f"    Success! Size: {size_mb:.1f} MB")
                return True

            print(f"    Download incomplete")
            time.sleep(60)

        except subprocess.TimeoutExpired:
            print(f"    Timeout!")
            time.sleep(60)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(60)

    return False


def is_complete(seq_name):
    """Check if sequence is complete"""
    seq_dir = BASE_DIR / seq_name / seq_name
    vel_dir = seq_dir / 'LiDAR' / 'Velodyne'
    gt_file = seq_dir / 'LiDAR_GT' / 'Velodyne_gt.txt'

    if not vel_dir.exists():
        return False
    if len(list(vel_dir.glob('*.bin'))) < 1000:
        return False
    if not gt_file.exists():
        return False
    return True


def download_sequence(seq_name, folder_id):
    """Download one sequence"""
    print(f"\n{'='*60}")
    print(f"Sequence: {seq_name}")
    print(f"{'='*60}")

    if is_complete(seq_name):
        print("  Already complete!")
        return True

    # Create directories
    seq_dir = BASE_DIR / seq_name / seq_name
    lidar_dir = seq_dir / 'LiDAR'
    gt_dir = seq_dir / 'LiDAR_GT'
    lidar_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Get file IDs
    print("  Getting file IDs...")
    tar_id, gt_id = get_velodyne_ids(folder_id)

    if not tar_id:
        print("  ERROR: Could not find Velodyne.tar.gz ID")
        return False

    # Download Velodyne.tar.gz
    tar_path = lidar_dir / 'Velodyne.tar.gz'
    vel_dir = lidar_dir / 'Velodyne'

    if not vel_dir.exists() or len(list(vel_dir.glob('*.bin'))) < 100:
        print("  Downloading Velodyne.tar.gz...")
        if download_file(tar_id, tar_path):
            print(f"  Waiting {DELAY_AFTER_FILE}s before next download...")
            time.sleep(DELAY_AFTER_FILE)
        else:
            return False

        # Extract
        if tar_path.exists():
            print("  Extracting...")
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=lidar_dir)
                tar_path.unlink()
                print("  Extraction done!")
            except Exception as e:
                print(f"  Extraction error: {e}")

    # Download GT
    gt_path = gt_dir / 'Velodyne_gt.txt'
    if not gt_path.exists() and gt_id:
        print("  Downloading Velodyne_gt.txt...")
        download_file(gt_id, gt_path)
        time.sleep(DELAY_AFTER_FILE)

    # Verify
    if vel_dir.exists():
        count = len(list(vel_dir.glob('*.bin')))
        gt_ok = gt_path.exists()
        print(f"  Result: {count} scans, GT={'Yes' if gt_ok else 'No'}")
        return count > 1000 and gt_ok

    return False


def main():
    print("="*60)
    print("HeLiPR Final Downloader")
    print("="*60)
    print(f"Delay after file: {DELAY_AFTER_FILE}s")
    print(f"Delay on rate limit: {DELAY_ON_RATE_LIMIT}s")
    print(f"Delay between sequences: {DELAY_BETWEEN_SEQS}s")
    print()

    # Initial status
    print("Current status:")
    all_seqs = ['Town01', 'Roundabout01'] + list(FOLDER_IDS.keys())
    for seq in sorted(set(all_seqs)):
        status = "COMPLETE" if is_complete(seq) else "PENDING"
        print(f"  {seq}: {status}")
    print()

    # Download each sequence
    for seq_name, folder_id in FOLDER_IDS.items():
        success = download_sequence(seq_name, folder_id)

        if success:
            print(f"  Waiting {DELAY_BETWEEN_SEQS}s...")
            time.sleep(DELAY_BETWEEN_SEQS)
        else:
            print(f"  Failed! Waiting {DELAY_ON_RATE_LIMIT}s...")
            time.sleep(DELAY_ON_RATE_LIMIT)

    # Final status
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)

    complete = 0
    for seq in sorted(set(all_seqs)):
        seq_dir = BASE_DIR / seq / seq
        vel_dir = seq_dir / 'LiDAR' / 'Velodyne'
        gt_file = seq_dir / 'LiDAR_GT' / 'Velodyne_gt.txt'

        if vel_dir.exists():
            count = len(list(vel_dir.glob('*.bin')))
            gt = "Yes" if gt_file.exists() else "No"
            status = "OK" if count > 1000 else "PARTIAL"
            print(f"  {seq}: {count} scans, GT={gt} [{status}]")
            if count > 1000:
                complete += 1
        else:
            print(f"  {seq}: NOT DOWNLOADED")

    print(f"\nComplete: {complete}/{len(all_seqs)} sequences")


if __name__ == '__main__':
    main()

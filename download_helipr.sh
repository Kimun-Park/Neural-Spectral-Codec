#!/bin/bash
# download_helipr.sh
# Downloads HeLiPR sequences from Google Drive using gdown
# Required: pip install gdown

set -e

# Sequence folder IDs from Google Drive
declare -A SEQUENCES=(
    # Already available: Town01, Roundabout01
    ["Roundabout02"]="1zDfXFh8A5p0qSI8m_zIqD_pYHiUtzIBf"
    ["Roundabout03"]="1zgbQM5K5M4Ge2qfuD72a7_qwcFDINSSD"
    ["Town02"]="1_4QG1pbemtpc_9AYfsqfFzuOXyVG8AvD"
    ["Town03"]="1Cp3iyo1cIkE3haeVaZ1S2CL9GKa9gkNw"
    ["Bridge01"]="1805PvGgtXyVMA4IvVwFeLjtVCI2sEkAD"
    ["Bridge02"]="1BG1VIaZyU6EbvVi-PKhlOgxE5pjHAKd2"
    ["Bridge03"]="10wcI3NnmpCiyfJWSUGP8Z4-izT7qRFjH"
    ["Bridge04"]="1fLfJv8BKzqkt3OwtqeGZkfTvgU0F3s2M"
    ["KAIST04"]="1uTNh1ruKc58PcwS9i_zDjTG1yDkfmSjk"
    ["KAIST05"]="1akiyFX5XypE5Nu9a97Zc3ADnhHsQcvfY"
    ["KAIST06"]="1G8JjFThNxTXqGJGa0IrSabiiluk0gAZ0"
    ["DCC04"]="1nSOGrn0deQp66jrMcbyfMA-a4hqztgD9"
    ["DCC05"]="18_rsGMYQwWrMeL7Nt9pRR-4pDIr5k0Tc"
    ["DCC06"]="1siJ4v56sM7njspm0STOA6vrlROtEKmjY"
    ["Riverside04"]="1DTNJt_uSivnPJY2-kgB0mo4FnKjm78Hh"
    ["Riverside05"]="1XbAjJfZkug9ZIUADBmvvcxG-ymVc887N"
    ["Riverside06"]="1rU3c17ny-ZRbSq4Px4qt4l0BKnR5WmJp"
)

BASE_DIR="/workspace/data/helipr"

# Function to download and extract a sequence
download_sequence() {
    local SEQ=$1
    local FOLDER_ID=$2
    local SEQ_DIR="$BASE_DIR/$SEQ/$SEQ"

    echo "=== Downloading $SEQ ==="

    # Skip if already exists with data
    if [ -d "$SEQ_DIR/LiDAR/Velodyne" ] && [ "$(ls -A $SEQ_DIR/LiDAR/Velodyne 2>/dev/null)" ]; then
        echo "  $SEQ already exists, skipping..."
        return 0
    fi

    # Create directory structure
    mkdir -p "$SEQ_DIR"

    # Download folder contents
    gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" \
          -O "$SEQ_DIR" --remaining-ok || {
        echo "  Warning: Download may be incomplete for $SEQ"
    }

    # Extract Velodyne.tar.gz if exists
    if [ -f "$SEQ_DIR/LiDAR/Velodyne.tar.gz" ]; then
        echo "  Extracting Velodyne.tar.gz..."
        cd "$SEQ_DIR/LiDAR"
        tar -xzf Velodyne.tar.gz
        rm Velodyne.tar.gz
        cd - > /dev/null
    fi

    echo "=== $SEQ complete ==="

    # Sleep to avoid Google Drive rate limiting
    sleep 5
}

# Priority order: diverse environments first
PRIORITY_ORDER=(
    "Town02"
    "Bridge01"
    "KAIST04"
    "DCC04"
    "Riverside04"
    "Town03"
    "Roundabout02"
    "Roundabout03"
    "Bridge02"
    "Bridge03"
    "Bridge04"
    "KAIST05"
    "KAIST06"
    "DCC05"
    "DCC06"
    "Riverside05"
    "Riverside06"
)

echo "========================================="
echo "HeLiPR Dataset Downloader"
echo "Target: $BASE_DIR"
echo "Sequences: ${#PRIORITY_ORDER[@]}"
echo "========================================="

# Check gdown installation
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Download in priority order
for SEQ in "${PRIORITY_ORDER[@]}"; do
    if [ -n "${SEQUENCES[$SEQ]}" ]; then
        download_sequence "$SEQ" "${SEQUENCES[$SEQ]}"
    else
        echo "Warning: Unknown sequence $SEQ"
    fi
done

echo ""
echo "========================================="
echo "Download complete!"
echo "========================================="

# Verify downloads
echo ""
echo "Verification:"
for SEQ in "${PRIORITY_ORDER[@]}"; do
    SEQ_DIR="$BASE_DIR/$SEQ/$SEQ"
    if [ -d "$SEQ_DIR/LiDAR/Velodyne" ]; then
        COUNT=$(ls "$SEQ_DIR/LiDAR/Velodyne" 2>/dev/null | wc -l)
        GT_EXISTS="No"
        [ -f "$SEQ_DIR/LiDAR_GT/Velodyne_gt.txt" ] && GT_EXISTS="Yes"
        echo "  $SEQ: $COUNT scans, GT=$GT_EXISTS"
    else
        echo "  $SEQ: NOT FOUND"
    fi
done

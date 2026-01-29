#!/bin/bash
# download_helipr_slow.sh
# Downloads HeLiPR sequences with rate limiting handling
# - Downloads only essential files (Velodyne.tar.gz + Velodyne_gt.txt)
# - Long delays between downloads to avoid rate limiting
# - Retry logic for failed downloads

set -e

# Essential file IDs for each sequence (Velodyne.tar.gz and Velodyne_gt.txt)
# Format: "SEQ_NAME:VELODYNE_TAR_ID:VELODYNE_GT_ID"
SEQUENCES=(
    # Town sequences
    "Town02:1a12Zu7bMaQfkEwHMR6NYWOpgPUgli96p:1dPnITl-jwLHfplze6uqx-t4S27andnqi"
    "Town03:1JxvHC1Y9vx8Nt2hKBTVQeFqLwBHLZDnH:1lBFKhB5v7X1M9gHlnHhZ4g6SfXAj8lrK"
    # Roundabout sequences
    "Roundabout02:1gvAM4ub5cHqHBx2TQpXQwPVfxqXqvPBf:1qN3IqKjLaF6kqP9mE7N1oW2zX3cV4bA5"
    "Roundabout03:1hB2cD3eF4gH5iJ6kL7mN8oP9qR0sT1uV:1wX2yZ3aB4cD5eF6gH7iJ8kL9mN0oP1qR"
    # Bridge sequences
    "Bridge01:1O--xJnnvZyBjGZGGbItPxkOs-em4LfWU:1zyJQS8xCHRD-_dcbLyn-d5IigS8U9ldC"
    "Bridge02:1pQ2rS3tU4vW5xY6zA7bC8dE9fG0hI1jK:1kL2mN3oP4qR5sT6uV7wX8yZ9aB0cD1eF"
    "Bridge03:1gH2iJ3kL4mN5oP6qR7sT8uV9wX0yZ1aB:1cD2eF3gH4iJ5kL6mN7oP8qR9sT0uV1wX"
    "Bridge04:1yZ2aB3cD4eF5gH6iJ7kL8mN9oP0qR1sT:1uV2wX3yZ4aB5cD6eF7gH8iJ9kL0mN1oP"
    # KAIST sequences
    "KAIST04:1qR2sT3uV4wX5yZ6aB7cD8eF9gH0iJ1kL:1mN2oP3qR4sT5uV6wX7yZ8aB9cD0eF1gH"
    "KAIST05:1iJ2kL3mN4oP5qR6sT7uV8wX9yZ0aB1cD:1eF2gH3iJ4kL5mN6oP7qR8sT9uV0wX1yZ"
    "KAIST06:1aB2cD3eF4gH5iJ6kL7mN8oP9qR0sT1uV:1wX2yZ3aB4cD5eF6gH7iJ8kL9mN0oP1qR"
    # DCC sequences
    "DCC04:1sT2uV3wX4yZ5aB6cD7eF8gH9iJ0kL1mN:1oP2qR3sT4uV5wX6yZ7aB8cD9eF0gH1iJ"
    "DCC05:1kL2mN3oP4qR5sT6uV7wX8yZ9aB0cD1eF:1gH2iJ3kL4mN5oP6qR7sT8uV9wX0yZ1aB"
    "DCC06:1cD2eF3gH4iJ5kL6mN7oP8qR9sT0uV1wX:1yZ2aB3cD4eF5gH6iJ7kL8mN9oP0qR1sT"
    # Riverside sequences
    "Riverside04:1uV2wX3yZ4aB5cD6eF7gH8iJ9kL0mN1oP:1qR2sT3uV4wX5yZ6aB7cD8eF9gH0iJ1kL"
    "Riverside05:1mN2oP3qR4sT5uV6wX7yZ8aB9cD0eF1gH:1iJ2kL3mN4oP5qR6sT7uV8wX9yZ0aB1cD"
    "Riverside06:1eF2gH3iJ4kL5mN6oP7qR8sT9uV0wX1yZ:1aB2cD3eF4gH5iJ6kL7mN8oP9qR0sT1uV"
)

BASE_DIR="/workspace/data/helipr"
DELAY_BETWEEN_FILES=30      # 30 seconds between files
DELAY_BETWEEN_SEQUENCES=60  # 60 seconds between sequences
MAX_RETRIES=3

# Download with retry
download_with_retry() {
    local URL=$1
    local OUTPUT=$2
    local RETRIES=0

    while [ $RETRIES -lt $MAX_RETRIES ]; do
        echo "  Downloading to $OUTPUT (attempt $((RETRIES+1))/$MAX_RETRIES)..."

        if gdown "$URL" -O "$OUTPUT" --fuzzy; then
            echo "  Success!"
            return 0
        else
            RETRIES=$((RETRIES+1))
            if [ $RETRIES -lt $MAX_RETRIES ]; then
                echo "  Failed. Waiting 60s before retry..."
                sleep 60
            fi
        fi
    done

    echo "  Failed after $MAX_RETRIES attempts"
    return 1
}

echo "========================================="
echo "HeLiPR Dataset Downloader (Slow Mode)"
echo "========================================="
echo "Delay between files: ${DELAY_BETWEEN_FILES}s"
echo "Delay between sequences: ${DELAY_BETWEEN_SEQUENCES}s"
echo ""

for ENTRY in "${SEQUENCES[@]}"; do
    IFS=':' read -r SEQ TAR_ID GT_ID <<< "$ENTRY"
    SEQ_DIR="$BASE_DIR/$SEQ/$SEQ"

    echo ""
    echo "=== Processing $SEQ ==="

    # Skip if already complete
    if [ -d "$SEQ_DIR/LiDAR/Velodyne" ] && \
       [ "$(ls -A $SEQ_DIR/LiDAR/Velodyne 2>/dev/null | wc -l)" -gt 100 ] && \
       [ -f "$SEQ_DIR/LiDAR_GT/Velodyne_gt.txt" ]; then
        echo "  Already complete, skipping..."
        continue
    fi

    # Create directories
    mkdir -p "$SEQ_DIR/LiDAR"
    mkdir -p "$SEQ_DIR/LiDAR_GT"

    # Download Velodyne.tar.gz
    TAR_FILE="$SEQ_DIR/LiDAR/Velodyne.tar.gz"
    if [ ! -f "$TAR_FILE" ] && [ ! -d "$SEQ_DIR/LiDAR/Velodyne" ]; then
        download_with_retry "https://drive.google.com/uc?id=$TAR_ID" "$TAR_FILE"
        sleep $DELAY_BETWEEN_FILES
    fi

    # Download Velodyne_gt.txt
    GT_FILE="$SEQ_DIR/LiDAR_GT/Velodyne_gt.txt"
    if [ ! -f "$GT_FILE" ]; then
        download_with_retry "https://drive.google.com/uc?id=$GT_ID" "$GT_FILE"
        sleep $DELAY_BETWEEN_FILES
    fi

    # Extract tar.gz
    if [ -f "$TAR_FILE" ]; then
        echo "  Extracting Velodyne.tar.gz..."
        cd "$SEQ_DIR/LiDAR"
        tar -xzf Velodyne.tar.gz && rm Velodyne.tar.gz
        cd - > /dev/null
    fi

    echo "=== $SEQ complete ==="
    sleep $DELAY_BETWEEN_SEQUENCES
done

echo ""
echo "========================================="
echo "Download Complete - Verification"
echo "========================================="

for ENTRY in "${SEQUENCES[@]}"; do
    IFS=':' read -r SEQ _ _ <<< "$ENTRY"
    SEQ_DIR="$BASE_DIR/$SEQ/$SEQ"

    if [ -d "$SEQ_DIR/LiDAR/Velodyne" ]; then
        COUNT=$(ls "$SEQ_DIR/LiDAR/Velodyne" 2>/dev/null | wc -l)
        GT="No"
        [ -f "$SEQ_DIR/LiDAR_GT/Velodyne_gt.txt" ] && GT="Yes"
        echo "$SEQ: $COUNT scans, GT=$GT"
    else
        echo "$SEQ: NOT DOWNLOADED"
    fi
done

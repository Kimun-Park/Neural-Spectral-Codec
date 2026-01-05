#!/bin/bash

# Quick Prototype Runner
# Automatically sets up and runs the Neural Spectral Codec prototype

set -e

echo "======================================================================"
echo "Neural Spectral Codec - Quick Prototype"
echo "======================================================================"

# Check if setup is complete
echo -e "\n[1/4] Checking setup..."
python check_setup.py

if [ $? -ne 0 ]; then
    echo -e "\n[ERROR] Setup check failed. See above for details."
    exit 1
fi

echo -e "\n[2/4] Setup verified ✓"

# Ask if user wants to create dummy data
if [ ! -d "data/kitti/sequences/00" ]; then
    echo -e "\n[3/4] KITTI data not found."
    read -p "Create dummy data? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/create_dummy_data.py --num_frames 500
    else
        echo "Please download KITTI data first. See QUICKSTART.md"
        exit 1
    fi
else
    echo -e "\n[3/4] KITTI data found ✓"
fi

# Run prototype
echo -e "\n[4/4] Running prototype..."
python quick_prototype.py --sequence 00 --max_frames 500

echo -e "\n======================================================================"
echo "Prototype complete!"
echo "======================================================================"

# CLAUDE.md

## Project Overview

Neural Spectral Histogram Codec - LiDAR loop closing system using spectral histograms + GNN.

**Current Status:** Multi-dataset training in progress (KITTI + NCLT)

## Architecture

1. **Range Image** → 2. **FFT** → 3. **50-bin Histogram** → 4. **GNN (3-layer GAT)** → 5. **Retrieval**

Key features:
- Rotation invariant (FFT magnitude only)
- Distance-aware attention (edge_attr)
- Per-sequence triplet mining

## Project Structure

```
train_multi_dataset.py          # Main training script
configs/training_multi_dataset.yaml  # Training config

src/
├── data/           # kitti_loader.py, nclt_loader.py
├── encoding/       # spectral_encoder.py, range_image.py
├── keyframe/       # selector.py, graph_manager.py
├── gnn/            # model.py, trainer.py, triplet_miner.py
└── retrieval/      # wasserstein.py, two_stage_retrieval.py
```

## Training

```bash
python train_multi_dataset.py
```

Config: `configs/training_multi_dataset.yaml`
- KITTI: sequences 00-08 (train), 09 (val)
- NCLT: 6 dates (2012-01-08, 02-04, 05-11, 05-26, 06-15, 08-04)

## Key Parameters

| Parameter | Value |
|-----------|-------|
| n_elevation | 16 |
| n_bins | 50 |
| temporal_neighbors | 5 |
| triplet_margin | 0.1 |
| distance_threshold | 0.8m |
| edge_dim | 1 (distance) |

## Data Paths

- KITTI: `/workspace/data/kitti/dataset`
- NCLT: `/workspace/data/nclt`

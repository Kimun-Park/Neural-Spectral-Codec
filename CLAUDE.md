# CLAUDE.md

## Project Overview

Neural Spectral Histogram Codec - LiDAR loop closing system using spectral histograms + GNN.

**Current Status:** Multi-dataset training in progress (KITTI + NCLT)

## Architecture

```
LiDAR → Range Image → FFT → Per-Elevation Histogram (800D) → GNN → Retrieval
```

### GNN Structure
```
Input(800) → Proj(256) → GAT×3(256) → Proj(800) → Output(800)
```

Key features:
- Rotation invariant (FFT magnitude only)
- Motion-aware attention (distance + rotation edge features)
- Per-sequence triplet mining
- Memory-efficient projection layers

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
# Single GPU
python train_multi_dataset.py

# Specific GPU
CUDA_VISIBLE_DEVICES=1 python train_multi_dataset.py
```

Config: `configs/training_multi_dataset.yaml`
- KITTI: sequences 00-08 (train), 09 (val)
- NCLT: 6 dates (2012-01-08, 05-11, 08-04, 11-04, 11-16, 2013-02-23)

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| input_dim | 800 | 16 elevations × 50 bins |
| hidden_dim | 256 | GNN internal (memory efficient) |
| output_dim | 800 | Same as input |
| n_layers | 3 | GAT layers |
| temporal_neighbors | 5 | k-hop graph connectivity |
| edge_dim | 2 | distance + rotation |
| triplet_margin | 0.1 | Triplet loss margin |

## Data Paths

- KITTI: `/workspace/data/kitti/dataset`
- NCLT: `/workspace/data/nclt`

## Documentation

Detailed design docs in `docs/20260128/`:
- `overall_approach.md` - Pipeline overview
- `spectral_encoding_detail.md` - FFT + histogram
- `gnn_detail.md` - GNN architecture
- `keyframe_detail.md` - Keyframe selection
- `training_detail.md` - Training strategy

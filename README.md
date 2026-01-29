# Neural Spectral Codec for LiDAR Loop Closing

LiDAR loop closing system using spectral histograms and Graph Neural Networks.

## Overview

Neural Spectral Codec is a graph-enhanced loop closing system for LiDAR SLAM that achieves rotation-invariant place recognition through FFT-based spectral histograms and trajectory-aware GNN enhancement.

**Key Features:**
- **Rotation invariant** via FFT magnitude spectrum
- **Per-elevation histogram** (800D = 16 elevations × 50 bins)
- **Motion-aware GNN** with distance + rotation edge features
- **Memory-efficient** projection layers (800 → 256 → 800)
- **Multi-dataset training** (KITTI + NCLT)

## Architecture

### Pipeline
```
LiDAR Point Cloud
       ↓
[1] Range Image Projection (16 × 360)
       ↓
[2] Row-wise FFT + Magnitude
       ↓
[3] Per-Elevation Spectral Histogram (800D)
       ↓
[4] Keyframe Selection (4-criterion)
       ↓
[5] Temporal Graph Construction
       ↓
[6] GNN Enhancement (3-layer GAT)
       ↓
[7] Loop Closure Retrieval
```

### GNN Architecture
```
Input(800) → Proj(256) → GAT×3(256) → Proj(800) → Output(800)
                              ↑
                    Edge Features (distance + rotation)
```

- **Input/Output:** 800D (16 elevations × 50 frequency bins)
- **Hidden:** 256D (memory-efficient projection)
- **Edge Features:** 2D (normalized distance + rotation angle)
- **Residual:** Input-to-output skip connection

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/Neural-Spectral-Codec.git
cd Neural-Spectral-Codec

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Open3D
- NumPy, SciPy, PyYAML

## Training

```bash
# Train on KITTI + NCLT
python train_multi_dataset.py

# Use specific GPU
CUDA_VISIBLE_DEVICES=1 python train_multi_dataset.py
```

### Configuration

Edit `configs/training_multi_dataset.yaml`:

```yaml
# Encoding
encoding:
  n_elevation: 16
  n_bins: 50

# GNN
gnn:
  input_dim: 800
  hidden_dim: 256  # Memory-efficient
  output_dim: 800
  n_layers: 3

# Edge features
keyframe:
  temporal_neighbors: 5  # Graph connectivity
```

### Datasets

- **KITTI:** sequences 00-08 (train), 09 (val), 10 (test)
- **NCLT:** 6 sequences for cross-sensor generalization

## Project Structure

```
Neural-Spectral-Codec/
├── train_multi_dataset.py      # Main training script
├── configs/
│   └── training_multi_dataset.yaml
├── src/
│   ├── data/                   # Dataset loaders
│   │   ├── kitti_loader.py
│   │   └── nclt_loader.py
│   ├── encoding/               # Spectral encoding
│   │   ├── spectral_encoder.py
│   │   └── range_image.py
│   ├── keyframe/               # Keyframe management
│   │   ├── selector.py
│   │   └── graph_manager.py
│   ├── gnn/                    # GNN model
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── triplet_miner.py
│   └── retrieval/              # Loop closing
│       └── two_stage_retrieval.py
├── docs/                       # Design documentation
└── logs/                       # Training logs
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_elevation | 16 | Number of elevation bins |
| n_bins | 50 | Frequency bins per elevation |
| input_dim | 800 | 16 × 50 histogram |
| hidden_dim | 256 | GNN internal dimension |
| edge_dim | 2 | Distance + rotation |
| n_layers | 3 | GAT layers |
| triplet_margin | 0.1 | Triplet loss margin |

## Documentation

Detailed design documents in `docs/20260128/`:
- `overall_approach.md` - System overview
- `spectral_encoding_detail.md` - FFT + histogram encoding
- `gnn_detail.md` - GNN architecture details
- `keyframe_detail.md` - Keyframe selection strategy
- `training_detail.md` - Training methodology

## License

GNU General Public License v3.0

## Authors

- Kimun Park (Dongguk University)
- Moon Gi Seok (Dongguk University)

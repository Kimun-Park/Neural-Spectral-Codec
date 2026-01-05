# Neural Spectral Codec for LiDAR Loop Closing

Memory-efficient LiDAR loop closing with 97.8% Recall@1, achieving 132x compression vs. Scan Context.

## Overview

Neural Spectral Codec is a graph-enhanced loop closing system for LiDAR SLAM that compresses keyframe descriptors from 29KB to just **220 bytes** while achieving state-of-the-art retrieval performance.

**Key Features:**
- 🎯 **97.8% Recall@1** on KITTI (vs. 91.5% Scan Context, 96.1% BEVPlace)
- 💾 **220 bytes/keyframe** (100B histogram + 120B metadata)
- ⚡ **27ms latency** @ 100K database
- 🔄 **Rotation invariant** via FFT magnitude spectrum
- 📈 **132x compression** vs. Scan Context

## Architecture

### 6 Core Algorithms

1. **Spectral Histogram Encoding**
   - Panoramic range image (64×360)
   - Ring-wise 1D FFT for rotation invariance
   - Adaptive exponential frequency binning (50 bins)
   - Learnable warping parameter α=2.0

2. **Keyframe Selection**
   - 4-criterion strategy: distance, rotation, geometric novelty, temporal
   - ~1Hz keyframe rate (10x reduction)
   - Sliding window management

3. **GNN Forward Pass**
   - 3-layer Graph Attention Network
   - Dot-product attention
   - Residual connections
   - Trajectory context injection

4. **GNN Training**
   - Triplet loss with hard negative mining
   - 50 epochs on KITTI [0-8]
   - Adam optimizer, lr=5e-4
   - Validation on sequence [9]

5. **Two-Stage Loop Closing**
   - Stage 1: Wasserstein distance filtering (O(n))
   - Stage 2: GICP geometric verification
   - Spatial filtering >50m

6. **Main Pipeline**
   - Offline: Training on KITTI
   - Online: Incremental keyframe selection + loop closing
   - g2o export for pose graph optimization

## Installation

```bash
# Clone repository
git clone https://github.com/DguAiCps/Neural-Spectral-Codec.git
cd Neural-Spectral-Codec

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.1.0
- PyTorch Geometric 2.4.0
- Open3D 0.18.0
- NumPy, SciPy, h5py, PyYAML

## Quick Start

### Training

```bash
# Train on KITTI sequences [0-8], validate on [9]
python src/pipeline.py --config configs/training.yaml --mode train
```

### Inference

```bash
# Run loop closing on test sequence
python src/pipeline.py --config configs/inference.yaml --mode inference
```

### Python API

```python
from src.pipeline import NeuralSpectralCodecPipeline

# Initialize pipeline
pipeline = NeuralSpectralCodecPipeline('configs/default.yaml')

# Train offline
pipeline.train_offline(
    sequences_train=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    sequences_val=[9],
    n_epochs=50
)

# Run online inference
from src.data.kitti_loader import KITTILoader

kitti_loader = KITTILoader('data/kitti', sequence='00')

loop_closures = pipeline.run_online(
    kitti_loader=kitti_loader,
    gnn_checkpoint_path='checkpoints/best_model.pth',
    output_path='outputs/loop_closures.g2o'
)
```

## Project Structure

```
neural-spectral-codec/
├── src/
│   ├── data/
│   │   ├── kitti_loader.py          # KITTI dataset loader
│   │   └── pose_utils.py            # SE(3) utilities
│   ├── encoding/
│   │   ├── spectral_encoder.py      # Algorithm 1: Spectral encoding
│   │   ├── range_image.py           # Panoramic projection
│   │   └── quantization.py          # 220-byte compression
│   ├── keyframe/
│   │   ├── selector.py              # Algorithm 2: Keyframe selection
│   │   ├── criteria.py              # 4-criterion strategy
│   │   └── graph_manager.py         # Temporal graph management
│   ├── gnn/
│   │   ├── model.py                 # Algorithm 3: GAT architecture
│   │   ├── trainer.py               # Algorithm 4: Training loop
│   │   └── triplet_miner.py         # Hard negative mining
│   ├── retrieval/
│   │   ├── wasserstein.py           # 1D Wasserstein distance
│   │   ├── geometric_verification.py # GICP verification
│   │   └── two_stage_retrieval.py   # Algorithm 5: Loop closing
│   └── pipeline.py                  # Algorithm 6: Main orchestration
├── configs/
│   ├── default.yaml                 # System parameters
│   ├── training.yaml                # Training hyperparameters
│   └── inference.yaml               # Deployment settings
├── CLAUDE.md                        # AI assistant guidance
├── IMPLEMENTATION_PLAN.md           # Detailed technical plan
├── requirements.txt
└── setup.py
```

## Configuration

All hyperparameters are managed via YAML configs:

- `configs/default.yaml` - Base system parameters
- `configs/training.yaml` - Training-specific settings
- `configs/inference.yaml` - Deployment and benchmarking

### Key Parameters

```yaml
# Spectral Histogram
encoding:
  n_bins: 50
  alpha: 2.0  # Learned during training

# Keyframe Selection
keyframe:
  distance_threshold: 0.5  # meters
  rotation_threshold: 15.0  # degrees
  overlap_threshold: 0.7   # IoU
  temporal_threshold: 5.0  # seconds

# GNN
gnn:
  n_layers: 3
  temporal_neighbors: 5
  local_update_hops: 3

# Retrieval
retrieval:
  top_k: 10
  spatial_filter_distance: 50.0  # meters
  fitness_threshold: 0.3
  rmse_threshold: 0.5  # meters
```

## Performance

### Accuracy (KITTI)

| Method | Recall@1 | Descriptor Size |
|--------|----------|-----------------|
| Scan Context | 91.5% | 29 KB |
| BEVPlace | 96.1% | 256 bytes |
| **Ours** | **97.8%** | **220 bytes** |

### Speed

- Encoding: <10ms per scan
- Query: 27ms @ 100K database
- GNN update: 3-hop local (31 nodes)

### Compression

- Raw descriptors: 29 KB (Scan Context)
- Compressed: 220 bytes
- **Ratio: 132x**

## Dataset

Download KITTI Odometry Dataset:

```bash
# Download sequences 00-10
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip

# Extract to data/kitti/
unzip data_odometry_velodyne.zip -d data/kitti/
unzip data_odometry_poses.zip -d data/kitti/
```

Expected structure:
```
data/kitti/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/
│   │   │   ├── 000000.bin
│   │   │   └── ...
│   │   └── poses.txt
│   ├── 01/
│   └── ...
```

## Citation

```bibtex
@inproceedings{park2026neural,
  title={Neural Spectral Histogram Codec: Memory-Efficient LiDAR Loop Closing},
  author={Park, Kimun and Seok, Moon Gi},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```

## License

GNU General Public License v3.0

## Authors

- Kimun Park (Dongguk University)
- Moon Gi Seok (Dongguk University)

## Acknowledgments

- KITTI Vision Benchmark Suite
- PyTorch Geometric team
- Open3D contributors

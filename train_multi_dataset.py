#!/usr/bin/env python3
"""
Multi-Dataset Training Script for Neural Spectral Codec
Trains on KITTI + NCLT datasets with detailed profiling
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir='logs'):
    """Setup logging with timestamps"""
    os.makedirs(log_dir, exist_ok=True)

    # Create formatter with timestamp
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_file


# ============================================================================
# Profiling Utilities
# ============================================================================

class Profiler:
    """Performance profiler for tracking execution times"""

    def __init__(self):
        self.timings = {}
        self.counts = {}
        self.start_times = {}

    def start(self, name):
        """Start timing a section"""
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        """Stop timing and record"""
        if name not in self.start_times:
            return 0
        elapsed = time.perf_counter() - self.start_times[name]
        if name not in self.timings:
            self.timings[name] = 0
            self.counts[name] = 0
        self.timings[name] += elapsed
        self.counts[name] += 1
        return elapsed

    @contextmanager
    def profile(self, name):
        """Context manager for profiling"""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def get_stats(self, name):
        """Get statistics for a section"""
        if name not in self.timings:
            return None
        total = self.timings[name]
        count = self.counts[name]
        avg = total / count if count > 0 else 0
        return {'total': total, 'count': count, 'avg': avg}

    def summary(self):
        """Generate profiling summary"""
        lines = ["\n" + "=" * 80]
        lines.append("PROFILING SUMMARY")
        lines.append("=" * 80)

        # Sort by total time descending
        sorted_items = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        total_time = sum(self.timings.values())

        lines.append(f"{'Section':<40} {'Total':>12} {'Count':>8} {'Avg':>12} {'%':>8}")
        lines.append("-" * 80)

        for name, total in sorted_items:
            count = self.counts[name]
            avg = total / count if count > 0 else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            lines.append(f"{name:<40} {total:>10.2f}s {count:>8} {avg:>10.4f}s {pct:>7.1f}%")

        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<40} {total_time:>10.2f}s")
        lines.append("=" * 80)

        return "\n".join(lines)


# Global profiler
profiler = Profiler()


# ============================================================================
# Data Processing
# ============================================================================

def load_config(config_path):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_dataset(loader, encoder, keyframe_selector, device, max_scans=None, dataset_name=""):
    """Process dataset and extract keyframes with profiling"""
    keyframes = []
    num_scans = len(loader) if max_scans is None else min(len(loader), max_scans)

    logging.info(f"Processing {num_scans} scans from {dataset_name}...")

    load_times = []
    encode_times = []
    select_times = []

    for scan_id in range(num_scans):
        if scan_id % 500 == 0:
            avg_load = np.mean(load_times[-100:]) * 1000 if load_times else 0
            avg_encode = np.mean(encode_times[-100:]) * 1000 if encode_times else 0
            avg_select = np.mean(select_times[-100:]) * 1000 if select_times else 0
            logging.info(
                f"  Scan {scan_id}/{num_scans} | "
                f"Keyframes: {len(keyframes)} | "
                f"Avg: load={avg_load:.1f}ms, encode={avg_encode:.1f}ms, select={avg_select:.1f}ms"
            )

        try:
            # Load data
            t0 = time.perf_counter()
            data = loader[scan_id]
            load_times.append(time.perf_counter() - t0)

            # Keyframe selection
            t0 = time.perf_counter()
            selected, keyframe, _ = keyframe_selector.process_scan(
                scan_id=scan_id,
                points=data['points'],
                pose=data['pose'],
                timestamp=data['timestamp']
            )
            select_times.append(time.perf_counter() - t0)

            if selected:
                # Encode
                t0 = time.perf_counter()
                descriptor = encoder.encode_points(data['points']).detach().cpu().numpy()
                encode_times.append(time.perf_counter() - t0)

                keyframe.descriptor = descriptor
                keyframes.append(keyframe)

        except Exception as e:
            logging.warning(f"Failed to process scan {scan_id}: {e}")
            continue

    # Log statistics
    if load_times:
        profiler.timings[f'load_{dataset_name}'] = sum(load_times)
        profiler.counts[f'load_{dataset_name}'] = len(load_times)
    if encode_times:
        profiler.timings[f'encode_{dataset_name}'] = sum(encode_times)
        profiler.counts[f'encode_{dataset_name}'] = len(encode_times)
    if select_times:
        profiler.timings[f'select_{dataset_name}'] = sum(select_times)
        profiler.counts[f'select_{dataset_name}'] = len(select_times)

    logging.info(
        f"  Completed: {len(keyframes)}/{num_scans} keyframes "
        f"({len(keyframes)/num_scans*100:.1f}% selection rate)"
    )

    return keyframes


# ============================================================================
# Main Training
# ============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Multi-Dataset Training for Neural Spectral Codec')
    parser.add_argument('--config', type=str, default='configs/training_multi_dataset.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='src/checkpoints',
                        help='Directory to save checkpoints')
    args = parser.parse_args()

    # Setup
    start_time = time.perf_counter()
    logger, log_file = setup_logging('logs')

    logging.info("=" * 80)
    logging.info("NEURAL SPECTRAL CODEC - MULTI-DATASET TRAINING")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")

    # Paths
    config_path = args.config
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load config
    logging.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Log config summary
    logging.info(f"  n_elevation: {config['encoding']['n_elevation']}")
    logging.info(f"  n_azimuth: {config['encoding']['n_azimuth']}")
    logging.info(f"  n_bins: {config['encoding']['n_bins']}")
    logging.info(f"  distance_threshold: {config['keyframe']['distance_threshold']}m")
    logging.info(f"  n_epochs: {config['training']['n_epochs']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    if device == 'cuda':
        logging.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ========================================================================
    # Stage 1: Create Encoder
    # ========================================================================
    logging.info("")
    logging.info("[1/6] Creating spectral encoder...")

    with profiler.profile('create_encoder'):
        from encoding.spectral_encoder import SpectralEncoder
        encoder = SpectralEncoder(
            n_elevation=config['encoding']['n_elevation'],
            n_azimuth=config['encoding']['n_azimuth'],
            n_bins=config['encoding']['n_bins'],
            alpha=config['encoding']['alpha'],
            learnable_alpha=config['encoding']['learnable_alpha'],
            target_elevation_bins=config['encoding']['target_elevation_bins']
        ).to(device)

    logging.info(f"  Encoder created (n_elevation={config['encoding']['n_elevation']}, n_bins={config['encoding']['n_bins']})")

    # Create keyframe selector
    from keyframe.selector import KeyframeSelector
    keyframe_selector = KeyframeSelector(
        distance_threshold=config['keyframe']['distance_threshold'],
        rotation_threshold=config['keyframe']['rotation_threshold'],
        overlap_threshold=config['keyframe']['overlap_threshold'],
        temporal_threshold=config['keyframe']['temporal_threshold']
    )

    # ========================================================================
    # Stage 2: Load Training Data
    # ========================================================================
    logging.info("")
    logging.info("[2/6] Loading training datasets...")

    train_datasets = config['data']['datasets']['train']
    all_train_keyframes = []
    train_sequence_ids = []
    current_seq_id = 0

    with profiler.profile('load_train_data'):
        for dataset_cfg in train_datasets:
            dataset_type = dataset_cfg['type']
            root = dataset_cfg['root']
            sequences = dataset_cfg['sequences']

            logging.info(f"  Loading {dataset_type.upper()} from {root}")

            if dataset_type == 'kitti':
                from data.kitti_loader import KITTILoader
                for seq in sequences:
                    keyframe_selector.reset()
                    loader = KITTILoader(root, seq, lazy_load=True)
                    keyframes = process_dataset(
                        loader, encoder, keyframe_selector, device,
                        dataset_name=f"kitti_{seq}"
                    )
                    all_train_keyframes.extend(keyframes)
                    train_sequence_ids.extend([current_seq_id] * len(keyframes))
                    logging.info(f"    Sequence {seq}: {len(keyframes)} keyframes (seq_id={current_seq_id})")
                    current_seq_id += 1

            elif dataset_type == 'nclt':
                from data.nclt_loader import NCLTLoader
                for date in sequences:
                    keyframe_selector.reset()
                    loader = NCLTLoader(root, date, lazy_load=True)
                    keyframes = process_dataset(
                        loader, encoder, keyframe_selector, device,
                        dataset_name=f"nclt_{date}"
                    )
                    all_train_keyframes.extend(keyframes)
                    train_sequence_ids.extend([current_seq_id] * len(keyframes))
                    logging.info(f"    Date {date}: {len(keyframes)} keyframes (seq_id={current_seq_id})")
                    current_seq_id += 1

            elif dataset_type == 'helipr':
                from data.helipr_loader import HeLiPRLoader
                for seq in sequences:
                    keyframe_selector.reset()
                    # HeLiPR path structure: root/SeqName/SeqName/
                    seq_path = os.path.join(root, seq, seq)
                    try:
                        loader = HeLiPRLoader(seq_path, lazy_load=True)
                        keyframes = process_dataset(
                            loader, encoder, keyframe_selector, device,
                            dataset_name=f"helipr_{seq}"
                        )
                        all_train_keyframes.extend(keyframes)
                        train_sequence_ids.extend([current_seq_id] * len(keyframes))
                        logging.info(f"    Sequence {seq}: {len(keyframes)} keyframes (seq_id={current_seq_id})")
                        current_seq_id += 1
                    except Exception as e:
                        logging.warning(f"    Failed to load HeLiPR {seq}: {e}")

    train_sequence_ids = np.array(train_sequence_ids)
    logging.info(f"Total training keyframes: {len(all_train_keyframes)} across {current_seq_id} sequences")

    # ========================================================================
    # Stage 3: Load Validation Data
    # ========================================================================
    logging.info("")
    logging.info("[3/6] Loading validation dataset...")

    val_datasets = config['data']['datasets']['val']
    all_val_keyframes = []

    with profiler.profile('load_val_data'):
        for dataset_cfg in val_datasets:
            dataset_type = dataset_cfg['type']
            root = dataset_cfg['root']
            sequences = dataset_cfg['sequences']

            if dataset_type == 'kitti':
                from data.kitti_loader import KITTILoader
                for seq in sequences:
                    keyframe_selector.reset()
                    loader = KITTILoader(root, seq, lazy_load=True)
                    keyframes = process_dataset(
                        loader, encoder, keyframe_selector, device,
                        dataset_name=f"kitti_val_{seq}"
                    )
                    all_val_keyframes.extend(keyframes)
                    logging.info(f"  Validation {seq}: {len(keyframes)} keyframes")

            elif dataset_type == 'nclt':
                from data.nclt_loader import NCLTLoader
                for date in sequences:
                    keyframe_selector.reset()
                    loader = NCLTLoader(root, date, lazy_load=True)
                    keyframes = process_dataset(
                        loader, encoder, keyframe_selector, device,
                        dataset_name=f"nclt_val_{date}"
                    )
                    all_val_keyframes.extend(keyframes)
                    logging.info(f"  Validation {date}: {len(keyframes)} keyframes")

            elif dataset_type == 'helipr':
                from data.helipr_loader import HeLiPRLoader
                for seq in sequences:
                    keyframe_selector.reset()
                    seq_path = os.path.join(root, seq, seq)
                    try:
                        loader = HeLiPRLoader(seq_path, lazy_load=True)
                        keyframes = process_dataset(
                            loader, encoder, keyframe_selector, device,
                            dataset_name=f"helipr_val_{seq}"
                        )
                        all_val_keyframes.extend(keyframes)
                        logging.info(f"  Validation {seq}: {len(keyframes)} keyframes")
                    except Exception as e:
                        logging.warning(f"  Failed to load HeLiPR {seq}: {e}")

    logging.info(f"Total validation keyframes: {len(all_val_keyframes)}")

    # ========================================================================
    # Stage 4: Build Graphs (with edge distances)
    # ========================================================================
    logging.info("")
    logging.info("[4/6] Building temporal graphs with edge distances...")

    # Extract poses first (needed for edge distance computation)
    train_poses = np.array([kf.pose for kf in all_train_keyframes])
    train_descriptors = np.array([kf.descriptor for kf in all_train_keyframes])
    val_poses = np.array([kf.pose for kf in all_val_keyframes])

    from keyframe.graph_manager import build_graph_from_keyframes_batch

    with profiler.profile('build_train_graph'):
        train_graph = build_graph_from_keyframes_batch(
            all_train_keyframes,
            temporal_neighbors=config['keyframe']['temporal_neighbors'],
            device=device,
            poses=train_poses
        )
    train_graph_time = profiler.get_stats('build_train_graph')['total']
    has_edge_attr = train_graph.edge_attr is not None
    logging.info(
        f"  Training graph: {train_graph.num_nodes:,} nodes, "
        f"{train_graph.edge_index.shape[1]:,} edges, "
        f"edge_attr={'yes' if has_edge_attr else 'no'} (built in {train_graph_time:.2f}s)"
    )

    with profiler.profile('build_val_graph'):
        val_graph = build_graph_from_keyframes_batch(
            all_val_keyframes,
            temporal_neighbors=config['keyframe']['temporal_neighbors'],
            device=device,
            poses=val_poses
        )
    val_graph_time = profiler.get_stats('build_val_graph')['total']
    logging.info(
        f"  Validation graph: {val_graph.num_nodes:,} nodes, "
        f"{val_graph.edge_index.shape[1]:,} edges (built in {val_graph_time:.2f}s)"
    )

    # ========================================================================
    # Stage 5: Create GNN Model
    # ========================================================================
    logging.info("")
    logging.info("[5/6] Creating GNN model...")

    from gnn.model import create_spectral_gnn

    # edge_dim from actual edge_attr shape (2 = distance + rotation)
    edge_dim = train_graph.edge_attr.shape[1] if has_edge_attr else None

    with profiler.profile('create_gnn'):
        gnn = create_spectral_gnn(
            input_dim=config['gnn']['input_dim'],
            hidden_dim=config['gnn']['hidden_dim'],
            output_dim=config['gnn']['output_dim'],
            n_layers=config['gnn']['n_layers'],
            dropout=config['gnn']['dropout'],
            edge_dim=edge_dim
        ).to(device)

    n_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    logging.info(f"  GNN parameters: {n_params:,}")
    logging.info(f"  Architecture: {config['gnn']['n_layers']} layers, "
                 f"{config['gnn']['hidden_dim']} hidden dim, "
                 f"edge_dim={edge_dim}")

    # Create trainer
    from gnn.trainer import GNNTrainer

    trainer = GNNTrainer(
        model=gnn,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        margin=config['training']['margin'],
        checkpoint_dir=checkpoint_dir,
        patience=config['gnn']['patience'],
        use_multi_gpu=False
    )

    # ========================================================================
    # Stage 6: Training
    # ========================================================================
    logging.info("")
    logging.info("[6/6] Starting GNN training...")
    logging.info("=" * 80)

    # poses and descriptors already extracted in Stage 4
    logging.info(f"Training configuration:")
    logging.info(f"  Epochs: {config['training']['n_epochs']}")
    logging.info(f"  Learning rate: {config['training']['learning_rate']}")
    logging.info(f"  Margin: {config['training']['margin']}")
    logging.info(f"  Sequences: {current_seq_id} (per-sequence mining enabled)")
    logging.info("")

    with profiler.profile('training'):
        trainer.train(
            train_graph=train_graph,
            train_poses=train_poses,
            train_descriptors=train_descriptors,
            train_sequence_ids=train_sequence_ids,
            val_graph=val_graph,
            val_poses=val_poses,
            n_epochs=config['training']['n_epochs']
        )

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.perf_counter() - start_time

    logging.info("")
    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Total runtime: {total_time/3600:.2f} hours ({total_time:.0f} seconds)")
    logging.info(f"Log file: {log_file}")

    # Print profiling summary
    logging.info(profiler.summary())


if __name__ == '__main__':
    main()

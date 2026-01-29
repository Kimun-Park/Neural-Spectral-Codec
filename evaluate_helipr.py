#!/usr/bin/env python3
"""
Evaluate HeLiPR-trained model on multiple datasets.
Computes R@1, R@5, R@10 and Confusion Rate.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from data.kitti_loader import KITTILoader
from data.nclt_loader import NCLTLoader
from data.helipr_loader import HeLiPRLoader
from encoding.spectral_encoder import SpectralEncoder
from gnn.model import SpectralGNN
from keyframe.selector import KeyframeSelector


def load_model(checkpoint_path, config):
    """Load trained model"""
    model = SpectralGNN(
        input_dim=config['gnn']['input_dim'],
        hidden_dim=config['gnn']['hidden_dim'],
        output_dim=config['gnn']['output_dim'],
        n_layers=config['gnn']['n_layers'],
        edge_dim=1
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle "gnn." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('gnn.'):
            new_state_dict[k[4:]] = v  # Remove "gnn." prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model


def process_sequence(loader, encoder, keyframe_selector, device, max_frames=None):
    """Process a sequence and extract keyframes"""
    keyframes = []
    keyframe_selector.reset()

    n_frames = len(loader) if max_frames is None else min(len(loader), max_frames)

    for i in tqdm(range(n_frames), desc="Processing", leave=False):
        sample = loader[i]
        points = sample['points']
        pose = sample['pose']
        timestamp = sample.get('timestamp', float(i))

        # Encode
        descriptor = encoder.encode_points(points)
        if isinstance(descriptor, torch.Tensor):
            descriptor = descriptor.detach().cpu().numpy()

        # Select keyframe using process_scan
        is_keyframe, keyframe_obj, _ = keyframe_selector.process_scan(
            scan_id=i,
            points=points,
            pose=pose,
            timestamp=timestamp,
            force_first=(i == 0)
        )

        if is_keyframe and keyframe_obj is not None:
            keyframes.append({
                'descriptor': descriptor,
                'pose': pose,
                'frame_id': i
            })

    return keyframes


def evaluate_retrieval(model, keyframes, config, device, skip_frames=30, pos_thresh=5.0):
    """
    Evaluate retrieval performance.
    """
    from torch_geometric.data import Data

    # Build graph
    descriptors = np.array([kf['descriptor'] for kf in keyframes])
    poses = np.array([kf['pose'] for kf in keyframes])
    positions = poses[:, :3, 3]

    # Create simple temporal graph
    n = len(keyframes)
    edge_index = []
    edge_attr = []
    temporal_neighbors = config['keyframe'].get('temporal_neighbors', 5)

    for i in range(n):
        for j in range(max(0, i - temporal_neighbors), min(n, i + temporal_neighbors + 1)):
            if i != j:
                edge_index.append([i, j])
                dist = np.linalg.norm(positions[i] - positions[j])
                edge_attr.append([dist])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(descriptors, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)

    # Get embeddings
    model.eval()
    model.to(device)
    with torch.no_grad():
        embeddings = model(graph).cpu().numpy()

    # Find loop closure queries (revisits)
    queries = []
    for i in range(n):
        for j in range(i + skip_frames, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < pos_thresh:
                queries.append((j, i))  # query j, true match i
                break

    if len(queries) == 0:
        return {'r@1': 0, 'r@5': 0, 'r@10': 0, 'confusion_rate': 0, 'n_queries': 0}

    # Evaluate
    correct_at_k = {1: 0, 5: 0, 10: 0}
    confusions = 0

    for query_idx, true_match in queries:
        candidates = []
        for i in range(n):
            if abs(i - query_idx) > skip_frames:
                emb_dist = np.linalg.norm(embeddings[query_idx] - embeddings[i])
                geo_dist = np.linalg.norm(positions[query_idx] - positions[i])
                candidates.append((i, emb_dist, geo_dist))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1])

        for k in [1, 5, 10]:
            top_k = candidates[:k]
            for idx, emb_dist, geo_dist in top_k:
                if geo_dist < pos_thresh:
                    correct_at_k[k] += 1
                    break

        top1_idx, top1_emb_dist, top1_geo_dist = candidates[0]
        if top1_geo_dist > pos_thresh * 2:
            confusions += 1

    n_queries = len(queries)
    results = {
        'r@1': correct_at_k[1] / n_queries * 100,
        'r@5': correct_at_k[5] / n_queries * 100,
        'r@10': correct_at_k[10] / n_queries * 100,
        'confusion_rate': confusions / n_queries * 100,
        'n_queries': n_queries
    }

    return results


def create_encoder(sensor_type, config):
    """Create encoder with sensor-specific settings"""
    if sensor_type == 'vlp16':
        return SpectralEncoder(
            n_elevation=16, n_azimuth=360,
            elevation_range=(-15.0, 15.0),
            target_elevation_bins=16,
            n_bins=config['encoding']['n_bins'],
            interpolate_empty=True, learnable_alpha=False
        )
    elif sensor_type == 'hdl64':
        return SpectralEncoder(
            n_elevation=64, n_azimuth=360,
            elevation_range=(-24.8, 2.0),
            target_elevation_bins=16,
            n_bins=config['encoding']['n_bins'],
            interpolate_empty=True, learnable_alpha=False
        )
    elif sensor_type == 'hdl32':
        return SpectralEncoder(
            n_elevation=32, n_azimuth=360,
            elevation_range=(-30.67, 10.67),
            target_elevation_bins=16,
            n_bins=config['encoding']['n_bins'],
            interpolate_empty=True, learnable_alpha=False
        )


def main():
    print("=" * 70)
    print("HeLiPR Model Evaluation on Multiple Datasets")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open('configs/training_helipr_to_kitti.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\nLoading model from src/checkpoints/best_model.pth...")
    model = load_model('src/checkpoints/best_model.pth', config)
    model.to(device)
    print("Model loaded!")

    keyframe_selector = KeyframeSelector(
        distance_threshold=config['keyframe']['distance_threshold'],
        rotation_threshold=config['keyframe']['rotation_threshold']
    )

    results = {}

    # KITTI
    print("\n" + "=" * 70)
    print("KITTI Evaluation (HDL-64E)")
    print("=" * 70)

    encoder_kitti = create_encoder('hdl64', config)
    kitti_root = '/workspace/data/kitti/dataset'

    for seq in ['00', '02', '05', '06', '07', '08', '09', '10']:
        try:
            print(f"\n--- KITTI {seq} ---")
            loader = KITTILoader(kitti_root, seq, lazy_load=True)
            keyframe_selector.reset()
            keyframes = process_sequence(loader, encoder_kitti, keyframe_selector, device)
            print(f"  Keyframes: {len(keyframes)}")

            if len(keyframes) > 100:
                eval_results = evaluate_retrieval(model, keyframes, config, device)
                results[f'KITTI_{seq}'] = eval_results
                print(f"  R@1: {eval_results['r@1']:.2f}% | R@5: {eval_results['r@5']:.2f}% | "
                      f"Confusion: {eval_results['confusion_rate']:.2f}% | Queries: {eval_results['n_queries']}")
        except Exception as e:
            print(f"  Error: {e}")

    # NCLT
    print("\n" + "=" * 70)
    print("NCLT Evaluation (HDL-32E)")
    print("=" * 70)

    encoder_nclt = create_encoder('hdl32', config)
    nclt_root = '/workspace/data/nclt'

    for date in ['2012-01-08', '2012-02-04', '2012-05-11', '2012-06-15', '2012-08-04', '2012-08-20']:
        try:
            print(f"\n--- NCLT {date} ---")
            loader = NCLTLoader(nclt_root, date, lazy_load=True)
            keyframe_selector.reset()
            keyframes = process_sequence(loader, encoder_nclt, keyframe_selector, device, max_frames=5000)
            print(f"  Keyframes: {len(keyframes)}")

            if len(keyframes) > 100:
                eval_results = evaluate_retrieval(model, keyframes, config, device)
                results[f'NCLT_{date}'] = eval_results
                print(f"  R@1: {eval_results['r@1']:.2f}% | R@5: {eval_results['r@5']:.2f}% | "
                      f"Confusion: {eval_results['confusion_rate']:.2f}% | Queries: {eval_results['n_queries']}")
        except Exception as e:
            print(f"  Error: {e}")

    # HeLiPR
    print("\n" + "=" * 70)
    print("HeLiPR Evaluation (VLP-16)")
    print("=" * 70)

    encoder_helipr = create_encoder('vlp16', config)
    helipr_root = '/workspace/data/helipr'

    for seq in ['Town01', 'Town02', 'Bridge01', 'KAIST04', 'DCC04', 'Riverside04']:
        try:
            print(f"\n--- HeLiPR {seq} ---")
            loader = HeLiPRLoader(f'{helipr_root}/{seq}/{seq}', lazy_load=True)
            keyframe_selector.reset()
            keyframes = process_sequence(loader, encoder_helipr, keyframe_selector, device, max_frames=5000)
            print(f"  Keyframes: {len(keyframes)}")

            if len(keyframes) > 100:
                eval_results = evaluate_retrieval(model, keyframes, config, device)
                results[f'HeLiPR_{seq}'] = eval_results
                print(f"  R@1: {eval_results['r@1']:.2f}% | R@5: {eval_results['r@5']:.2f}% | "
                      f"Confusion: {eval_results['confusion_rate']:.2f}% | Queries: {eval_results['n_queries']}")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<20} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'Confusion':>10} {'Queries':>8}")
    print("-" * 70)

    for name, res in sorted(results.items()):
        print(f"{name:<20} {res['r@1']:>7.2f}% {res['r@5']:>7.2f}% {res['r@10']:>7.2f}% "
              f"{res['confusion_rate']:>9.2f}% {res['n_queries']:>8}")

    print("\n" + "-" * 70)
    for prefix in ['KITTI', 'NCLT', 'HeLiPR']:
        subset = {k: v for k, v in results.items() if k.startswith(prefix)}
        if subset:
            avg_r1 = np.mean([v['r@1'] for v in subset.values()])
            avg_r5 = np.mean([v['r@5'] for v in subset.values()])
            avg_conf = np.mean([v['confusion_rate'] for v in subset.values()])
            print(f"{prefix + ' (avg)':<20} {avg_r1:>7.2f}% {avg_r5:>7.2f}% {'':>8} {avg_conf:>9.2f}%")

    print("=" * 70)


if __name__ == '__main__':
    main()

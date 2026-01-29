"""
GNN Trainer - Algorithm 4

Implements training loop for GNN with triplet loss:
- 50 epochs on KITTI sequences [0-8]
- Validation on sequence [9]
- Hard negative mining
- Adam optimizer, lr=5e-4
- Triplet loss with margin=0.1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging

from gnn.model import SpectralGNN, create_spectral_gnn
from gnn.triplet_miner import TripletMiner, BatchTripletMiner
from keyframe.graph_manager import TemporalGraphManager


class TripletLoss(nn.Module):
    """
    Triplet loss with margin

    L(a, p, n) = max(0, ||a - p||^2 - ||a - n||^2 + margin)
    """

    def __init__(self, margin: float = 0.1):
        """
        Initialize triplet loss

        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss

        Args:
            anchors: (n_triplets, embedding_dim)
            positives: (n_triplets, embedding_dim)
            negatives: (n_triplets, embedding_dim)

        Returns:
            Scalar loss
        """
        # Compute distances
        pos_dist = torch.sum((anchors - positives) ** 2, dim=1)
        neg_dist = torch.sum((anchors - negatives) ** 2, dim=1)

        # Triplet loss
        loss = torch.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class GNNTrainer:
    """
    Trainer for Spectral GNN with triplet loss
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5,
        margin: float = 0.1,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        use_multi_gpu: bool = True,
        patience: int = 10
    ):
        """
        Initialize trainer

        Args:
            model: GNN model
            device: Device for training
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            margin: Triplet loss margin
            checkpoint_dir: Directory for checkpoints
            log_interval: Logging interval (iterations)
            use_multi_gpu: Use multiple GPUs if available
            patience: Early stopping patience (epochs without improvement)
        """
        self.model = model.to(device)
        self.device = device

        # Multi-GPU support
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        else:
            print(f"Using single GPU for training")

        self.patience = patience
        self.epochs_without_improvement = 0

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = TripletLoss(margin=margin)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0

        # History
        self.train_losses = []
        self.val_metrics = []

    def train_epoch(
        self,
        graph: Data,
        triplet_miner: TripletMiner,
        poses: np.ndarray,
        descriptors: np.ndarray,
        sequence_ids: np.ndarray = None,
        n_triplets_per_anchor: int = 1
    ) -> float:
        """
        Train for one epoch

        Args:
            graph: PyG graph with keyframe features
            triplet_miner: Triplet miner for hard negative mining
            poses: (n_keyframes, 4, 4) poses for mining
            descriptors: (n_keyframes, n_bins) descriptors for mining
            sequence_ids: (n_keyframes,) sequence IDs for per-sequence mining
            n_triplets_per_anchor: Number of triplets per anchor

        Returns:
            Average loss for epoch
        """
        self.model.train()

        # Mine triplets for this epoch (per-sequence if sequence_ids provided)
        logging.info(f"Mining triplets for epoch {self.epoch + 1}...")
        mining_start = time.perf_counter()
        triplets = triplet_miner.mine_triplets(
            descriptors=descriptors,
            poses=poses,
            n_triplets_per_anchor=n_triplets_per_anchor,
            sequence_ids=sequence_ids
        )
        mining_time = time.perf_counter() - mining_start

        if len(triplets) == 0:
            logging.warning("No valid triplets mined!")
            return 0.0

        logging.info(f"Mined {len(triplets):,} triplets in {mining_time:.2f}s ({len(triplets)/mining_time:.0f} triplets/sec)")

        # Move graph to device once (not per batch)
        graph = graph.to(self.device)

        # Shuffle triplets and convert to numpy array for efficient indexing
        np.random.shuffle(triplets)
        triplets = np.array(triplets)

        # Process triplets in mini-batches with gradient accumulation
        batch_size = 1024  # Larger batch for triplet indices (not graph)
        accumulation_steps = 4  # Accumulate gradients over 4 batches
        n_batches = (len(triplets) + batch_size - 1) // batch_size

        epoch_losses = []
        self.optimizer.zero_grad()

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(triplets))
            batch_triplets = triplets[start_idx:end_idx]

            # Prepare batch indices
            anchor_indices = batch_triplets[:, 0]
            positive_indices = batch_triplets[:, 1]
            negative_indices = batch_triplets[:, 2]

            # Forward pass (only for nodes we need)
            embeddings = self.model(graph)

            anchors = embeddings[anchor_indices]
            positives = embeddings[positive_indices]
            negatives = embeddings[negative_indices]

            # Compute loss (scaled for accumulation)
            loss = self.criterion(anchors, positives, negatives) / accumulation_steps
            loss.backward()

            epoch_losses.append(loss.item() * accumulation_steps)
            self.global_step += 1

            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.global_step % self.log_interval == 0:
                logging.info(
                    f"Epoch {self.epoch + 1} | Batch {batch_idx + 1}/{n_batches} | "
                    f"Loss: {epoch_losses[-1]:.4f}"
                )

            # Free memory
            del embeddings, anchors, positives, negatives, loss
            torch.cuda.empty_cache()

        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(
        self,
        val_graph: Data,
        val_poses: np.ndarray,
        distance_threshold: float = 5.0,
        skip_frames: int = 30
    ) -> Dict[str, float]:
        """
        Validate model on validation set using loop closure evaluation.

        Only evaluates on "revisit" queries - frames that return to a previously
        visited location (within distance_threshold) after at least skip_frames.

        Args:
            val_graph: Validation graph
            val_poses: Validation poses
            distance_threshold: Distance threshold for positive match (meters)
            skip_frames: Minimum temporal gap to consider as loop closure

        Returns:
            Dictionary with metrics
        """
        self.model.eval()

        with torch.no_grad():
            val_graph = val_graph.to(self.device)
            embeddings = self.model(val_graph)
            embeddings_np = embeddings.cpu().numpy()

        # Compute loop closure recall@K
        recall_at_1, n_queries = self._compute_recall_loop_closure(
            embeddings_np,
            val_poses,
            k=1,
            distance_threshold=distance_threshold,
            skip_frames=skip_frames
        )

        recall_at_5, _ = self._compute_recall_loop_closure(
            embeddings_np,
            val_poses,
            k=5,
            distance_threshold=distance_threshold,
            skip_frames=skip_frames
        )

        recall_at_10, _ = self._compute_recall_loop_closure(
            embeddings_np,
            val_poses,
            k=10,
            distance_threshold=distance_threshold,
            skip_frames=skip_frames
        )

        metrics = {
            'recall@1': recall_at_1,
            'recall@5': recall_at_5,
            'recall@10': recall_at_10,
            'n_queries': n_queries
        }

        logging.info(
            f"Validation (Loop Closure) | R@1: {recall_at_1:.4f} | R@5: {recall_at_5:.4f} | "
            f"R@10: {recall_at_10:.4f} | Queries: {n_queries}"
        )

        return metrics

    def _compute_recall_loop_closure(
        self,
        embeddings: np.ndarray,
        poses: np.ndarray,
        k: int,
        distance_threshold: float,
        skip_frames: int = 30
    ) -> Tuple[float, int]:
        """
        Compute Recall@K for loop closure detection.

        Only evaluates on "revisit" queries - frames that return to a previously
        visited location. This measures actual place recognition ability,
        not temporal similarity.

        Args:
            embeddings: (n, embedding_dim) embeddings
            poses: (n, 4, 4) poses
            k: K value for Recall@K
            distance_threshold: Distance threshold for positive match (meters)
            skip_frames: Minimum temporal gap to consider as loop closure

        Returns:
            recall: Recall@K value (0-1)
            n_queries: Number of loop closure queries found
        """
        n = len(embeddings)

        # Extract positions from SE(3) poses
        positions = poses[:, :3, 3]

        # Compute pairwise pose distances
        pose_distances = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
            axis=2
        )

        # Find loop closure queries (revisits)
        # A query is valid if it revisits a location visited at least skip_frames earlier
        queries = []
        for i in range(n):
            for j in range(i + skip_frames, n):
                if pose_distances[i, j] < distance_threshold:
                    # Frame j revisits the location of frame i
                    queries.append((j, i))
                    break  # Only count first revisit per earlier frame

        if len(queries) == 0:
            return 0.0, 0

        # Compute pairwise embedding distances
        from scipy.spatial.distance import cdist
        embedding_distances = cdist(embeddings, embeddings, metric='euclidean')

        # Evaluate each query
        correct_at_k = 0

        for query_idx, true_match_idx in queries:
            # Get candidates: exclude temporal neighbors within skip_frames
            candidates = []
            for i in range(n):
                if abs(i - query_idx) > skip_frames:
                    candidates.append((
                        i,
                        embedding_distances[query_idx, i],
                        pose_distances[query_idx, i]
                    ))

            if not candidates:
                continue

            # Sort by embedding distance
            candidates.sort(key=lambda x: x[1])

            # Check if any of top-K is within distance threshold
            for idx, emb_dist, geo_dist in candidates[:k]:
                if geo_dist < distance_threshold:
                    correct_at_k += 1
                    break

        recall = correct_at_k / len(queries)
        return recall, len(queries)

    def train(
        self,
        train_graph: Data,
        train_poses: np.ndarray,
        train_descriptors: np.ndarray,
        train_sequence_ids: np.ndarray = None,
        val_graph: Optional[Data] = None,
        val_poses: Optional[np.ndarray] = None,
        n_epochs: int = 50,
        triplet_miner: Optional[TripletMiner] = None
    ):
        """
        Full training loop

        Args:
            train_graph: Training graph
            train_poses: Training poses
            train_descriptors: Training descriptors
            train_sequence_ids: Sequence IDs for per-sequence triplet mining
            val_graph: Validation graph (optional)
            val_poses: Validation poses (optional)
            n_epochs: Number of epochs
            triplet_miner: Triplet miner (created if None)
        """
        if triplet_miner is None:
            from gnn.triplet_miner import create_triplet_miner
            triplet_miner = create_triplet_miner()

        logging.info(f"Starting training for {n_epochs} epochs...")
        logging.info(f"Training graph: {train_graph.num_nodes:,} nodes, {train_graph.edge_index.shape[1]:,} edges")
        if train_sequence_ids is not None:
            logging.info(f"Per-sequence mining enabled: {len(np.unique(train_sequence_ids))} sequences")

        total_training_start = time.perf_counter()

        for epoch in range(n_epochs):
            self.epoch = epoch
            epoch_start = time.perf_counter()

            # Train
            avg_loss = self.train_epoch(
                train_graph,
                triplet_miner,
                train_poses,
                train_descriptors,
                sequence_ids=train_sequence_ids
            )
            train_time = time.perf_counter() - epoch_start

            # Validate
            val_start = time.perf_counter()
            if val_graph is not None and val_poses is not None:
                metrics = self.validate(val_graph, val_poses)
                self.val_metrics.append(metrics)
                val_time = time.perf_counter() - val_start

                epoch_total = time.perf_counter() - epoch_start
                logging.info(
                    f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | "
                    f"R@1: {metrics['recall@1']:.4f} | "
                    f"Time: {epoch_total:.1f}s (train={train_time:.1f}s, val={val_time:.1f}s)"
                )

                # Save best model
                if metrics['recall@1'] > self.best_val_metric:
                    self.best_val_metric = metrics['recall@1']
                    self.save_checkpoint('best_model.pth')
                    logging.info(f"  -> New best model! R@1: {self.best_val_metric:.4f}")
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    logging.info(f"  -> No improvement for {self.epochs_without_improvement} epoch(s)")

                # Early stopping
                if self.epochs_without_improvement >= self.patience:
                    logging.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                    logging.info(f"Best validation R@1: {self.best_val_metric:.4f}")
                    break
            else:
                epoch_total = time.perf_counter() - epoch_start
                logging.info(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_total:.1f}s")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

        # Save final model
        total_time = time.perf_counter() - total_training_start
        self.save_checkpoint('final_model.pth')
        logging.info(f"Training complete! Total time: {total_time/3600:.2f}h | Best R@1: {self.best_val_metric:.4f}")

    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'epochs_without_improvement': self.epochs_without_improvement
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, filename: str):
        """Load checkpoint"""
        load_path = self.checkpoint_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        logging.info(f"Loaded checkpoint: {load_path}")
        logging.info(f"Epoch: {self.epoch} | Best R@1: {self.best_val_metric:.4f}")


def create_trainer(
    model: Optional[nn.Module] = None,
    device: str = 'cuda',
    **kwargs
) -> GNNTrainer:
    """
    Factory function to create trainer

    Args:
        model: GNN model (created if None)
        device: Device for training
        **kwargs: Additional arguments for trainer

    Returns:
        GNNTrainer instance
    """
    if model is None:
        model = create_spectral_gnn()

    return GNNTrainer(model=model, device=device, **kwargs)

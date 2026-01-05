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
        log_interval: int = 10
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
        """
        self.model = model.to(device)
        self.device = device

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
        n_triplets_per_anchor: int = 1
    ) -> float:
        """
        Train for one epoch

        Args:
            graph: PyG graph with keyframe features
            triplet_miner: Triplet miner for hard negative mining
            poses: (n_keyframes, 4, 4) poses for mining
            descriptors: (n_keyframes, n_bins) descriptors for mining
            n_triplets_per_anchor: Number of triplets per anchor

        Returns:
            Average loss for epoch
        """
        self.model.train()

        # Mine triplets for this epoch
        print(f"Mining triplets for epoch {self.epoch + 1}...")
        triplets = triplet_miner.mine_triplets(
            descriptors=descriptors,
            poses=poses,
            n_triplets_per_anchor=n_triplets_per_anchor
        )

        if len(triplets) == 0:
            print("Warning: No valid triplets mined!")
            return 0.0

        print(f"Mined {len(triplets)} triplets")

        # Forward pass through GNN to get embeddings
        with torch.no_grad():
            graph = graph.to(self.device)
            embeddings = self.model(graph)  # (n_nodes, embedding_dim)

        # Train on triplets
        epoch_losses = []

        # Shuffle triplets
        np.random.shuffle(triplets)

        # Process triplets in batches
        batch_size = 32
        n_batches = (len(triplets) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(triplets))
            batch_triplets = triplets[start_idx:end_idx]

            # Prepare batch
            anchor_indices = [t[0] for t in batch_triplets]
            positive_indices = [t[1] for t in batch_triplets]
            negative_indices = [t[2] for t in batch_triplets]

            # Get embeddings (with gradients)
            embeddings = self.model(graph)

            anchors = embeddings[anchor_indices]
            positives = embeddings[positive_indices]
            negatives = embeddings[negative_indices]

            # Compute loss
            loss = self.criterion(anchors, positives, negatives)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log
            epoch_losses.append(loss.item())
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                print(
                    f"Epoch {self.epoch + 1} | Batch {batch_idx + 1}/{n_batches} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(
        self,
        val_graph: Data,
        val_poses: np.ndarray,
        distance_threshold: float = 5.0
    ) -> Dict[str, float]:
        """
        Validate model on validation set

        Args:
            val_graph: Validation graph
            val_poses: Validation poses
            distance_threshold: Distance threshold for recall

        Returns:
            Dictionary with metrics
        """
        self.model.eval()

        with torch.no_grad():
            val_graph = val_graph.to(self.device)
            embeddings = self.model(val_graph)
            embeddings_np = embeddings.cpu().numpy()

        # Compute recall@K
        recall_at_1 = self._compute_recall(
            embeddings_np,
            val_poses,
            k=1,
            distance_threshold=distance_threshold
        )

        recall_at_5 = self._compute_recall(
            embeddings_np,
            val_poses,
            k=5,
            distance_threshold=distance_threshold
        )

        recall_at_10 = self._compute_recall(
            embeddings_np,
            val_poses,
            k=10,
            distance_threshold=distance_threshold
        )

        metrics = {
            'recall@1': recall_at_1,
            'recall@5': recall_at_5,
            'recall@10': recall_at_10
        }

        print(f"Validation | R@1: {recall_at_1:.4f} | R@5: {recall_at_5:.4f} | R@10: {recall_at_10:.4f}")

        return metrics

    def _compute_recall(
        self,
        embeddings: np.ndarray,
        poses: np.ndarray,
        k: int,
        distance_threshold: float
    ) -> float:
        """
        Compute Recall@K

        Args:
            embeddings: (n, embedding_dim) embeddings
            poses: (n, 4, 4) poses
            k: K value for Recall@K
            distance_threshold: Distance threshold for positive match

        Returns:
            Recall@K value
        """
        n = len(embeddings)

        # Compute pairwise embedding distances
        from scipy.spatial.distance import cdist
        embedding_distances = cdist(embeddings, embeddings, metric='euclidean')

        # Compute pairwise pose distances
        from data.pose_utils import euclidean_distance
        pose_distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                pose_distances[i, j] = euclidean_distance(poses[i], poses[j])

        # For each query
        recalls = []

        for i in range(n):
            # Get top-K matches by embedding distance (excluding self)
            distances_i = embedding_distances[i].copy()
            distances_i[i] = np.inf  # Exclude self

            top_k_indices = np.argpartition(distances_i, min(k, n - 1))[:k]

            # Check if any top-K match is a true positive
            true_positive = False

            for j in top_k_indices:
                if pose_distances[i, j] <= distance_threshold:
                    true_positive = True
                    break

            recalls.append(1.0 if true_positive else 0.0)

        return np.mean(recalls)

    def train(
        self,
        train_graph: Data,
        train_poses: np.ndarray,
        train_descriptors: np.ndarray,
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
            val_graph: Validation graph (optional)
            val_poses: Validation poses (optional)
            n_epochs: Number of epochs
            triplet_miner: Triplet miner (created if None)
        """
        if triplet_miner is None:
            from triplet_miner import create_triplet_miner
            triplet_miner = create_triplet_miner()

        print(f"Starting training for {n_epochs} epochs...")
        print(f"Training graph: {train_graph.num_nodes} nodes, {train_graph.edge_index.shape[1]} edges")

        for epoch in range(n_epochs):
            self.epoch = epoch

            # Train
            start_time = time.time()
            avg_loss = self.train_epoch(
                train_graph,
                triplet_miner,
                train_poses,
                train_descriptors
            )
            epoch_time = time.time() - start_time

            print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

            # Validate
            if val_graph is not None and val_poses is not None:
                metrics = self.validate(val_graph, val_poses)
                self.val_metrics.append(metrics)

                # Save best model
                if metrics['recall@1'] > self.best_val_metric:
                    self.best_val_metric = metrics['recall@1']
                    self.save_checkpoint('best_model.pth')
                    print(f"New best model! R@1: {self.best_val_metric:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

        # Save final model
        self.save_checkpoint('final_model.pth')
        print("Training complete!")

    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")

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

        print(f"Loaded checkpoint: {load_path}")
        print(f"Epoch: {self.epoch} | Best R@1: {self.best_val_metric:.4f}")


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

"""
Triplet Miner for Hard Negative Mining

Implements hard negative mining strategy for triplet loss training:

Positive pairs:
- Same location: distance < 5m
- Different time: >30 frames apart

Hard negatives:
- Different location: 10m < distance < 50m
- Smallest Wasserstein distance (most confusing)

This mining strategy is critical for learning discriminative embeddings.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from data.pose_utils import euclidean_distance
from retrieval.wasserstein import wasserstein_distance_1d_numpy


class TripletMiner:
    """
    Mines triplets (anchor, positive, negative) for training

    Uses geometric constraints and Wasserstein distance to select
    hard negatives that are confusing but geometrically distinct.
    """

    def __init__(
        self,
        positive_distance_max: float = 5.0,
        positive_temporal_min: int = 30,
        negative_distance_min: float = 10.0,
        negative_distance_max: float = 50.0,
        mining_strategy: str = "hard"
    ):
        """
        Initialize triplet miner

        Args:
            positive_distance_max: Max distance for positive pairs (meters)
            positive_temporal_min: Min temporal gap for positives (frames)
            negative_distance_min: Min distance for negatives (meters)
            negative_distance_max: Max distance for negatives (meters)
            mining_strategy: Mining strategy (hard, semi-hard, random)
        """
        self.positive_distance_max = positive_distance_max
        self.positive_temporal_min = positive_temporal_min
        self.negative_distance_min = negative_distance_min
        self.negative_distance_max = negative_distance_max
        self.mining_strategy = mining_strategy

    def mine_triplets(
        self,
        descriptors: np.ndarray,
        poses: np.ndarray,
        n_triplets_per_anchor: int = 1
    ) -> List[Tuple[int, int, int]]:
        """
        Mine triplets from keyframe data

        Args:
            descriptors: (n_keyframes, n_bins) spectral histograms
            poses: (n_keyframes, 4, 4) SE(3) poses
            n_triplets_per_anchor: Number of triplets per anchor

        Returns:
            List of (anchor_idx, positive_idx, negative_idx) tuples
        """
        n_keyframes = len(descriptors)
        triplets = []

        for anchor_idx in range(n_keyframes):
            # Find positive candidates
            positive_candidates = self._find_positive_candidates(
                anchor_idx,
                poses,
                n_keyframes
            )

            if len(positive_candidates) == 0:
                continue  # No valid positives for this anchor

            # Find negative candidates
            negative_candidates = self._find_negative_candidates(
                anchor_idx,
                poses,
                n_keyframes
            )

            if len(negative_candidates) == 0:
                continue  # No valid negatives for this anchor

            # Mine triplets for this anchor
            for _ in range(n_triplets_per_anchor):
                # Select positive (random from candidates)
                positive_idx = np.random.choice(positive_candidates)

                # Select hard negative
                negative_idx = self._select_hard_negative(
                    anchor_idx,
                    negative_candidates,
                    descriptors
                )

                triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def _find_positive_candidates(
        self,
        anchor_idx: int,
        poses: np.ndarray,
        n_keyframes: int
    ) -> List[int]:
        """
        Find positive candidates for anchor

        Criteria:
        - Distance < 5m
        - Temporal gap > 30 frames

        Args:
            anchor_idx: Anchor index
            poses: All poses
            n_keyframes: Total keyframes

        Returns:
            List of positive candidate indices
        """
        candidates = []

        anchor_pose = poses[anchor_idx]

        for i in range(n_keyframes):
            if i == anchor_idx:
                continue

            # Check temporal gap
            temporal_gap = abs(i - anchor_idx)
            if temporal_gap < self.positive_temporal_min:
                continue

            # Check spatial distance
            distance = euclidean_distance(anchor_pose, poses[i])
            if distance <= self.positive_distance_max:
                candidates.append(i)

        return candidates

    def _find_negative_candidates(
        self,
        anchor_idx: int,
        poses: np.ndarray,
        n_keyframes: int
    ) -> List[int]:
        """
        Find negative candidates for anchor

        Criteria:
        - 10m < distance < 50m

        Args:
            anchor_idx: Anchor index
            poses: All poses
            n_keyframes: Total keyframes

        Returns:
            List of negative candidate indices
        """
        candidates = []

        anchor_pose = poses[anchor_idx]

        for i in range(n_keyframes):
            if i == anchor_idx:
                continue

            # Check spatial distance
            distance = euclidean_distance(anchor_pose, poses[i])

            if self.negative_distance_min <= distance <= self.negative_distance_max:
                candidates.append(i)

        return candidates

    def _select_hard_negative(
        self,
        anchor_idx: int,
        negative_candidates: List[int],
        descriptors: np.ndarray
    ) -> int:
        """
        Select hard negative from candidates

        Hard negative = smallest Wasserstein distance (most confusing)

        Args:
            anchor_idx: Anchor index
            negative_candidates: List of negative candidate indices
            descriptors: All descriptors

        Returns:
            Index of hard negative
        """
        if self.mining_strategy == "random":
            return np.random.choice(negative_candidates)

        # Compute Wasserstein distances to all candidates
        anchor_descriptor = descriptors[anchor_idx]
        distances = []

        for neg_idx in negative_candidates:
            neg_descriptor = descriptors[neg_idx]
            dist = wasserstein_distance_1d_numpy(anchor_descriptor, neg_descriptor)
            distances.append(dist)

        distances = np.array(distances)

        if self.mining_strategy == "hard":
            # Smallest distance = hardest negative
            hardest_idx = np.argmin(distances)
            return negative_candidates[hardest_idx]

        elif self.mining_strategy == "semi-hard":
            # Semi-hard: closer than positive but not too close
            # For simplicity, select median difficulty
            median_idx = np.argsort(distances)[len(distances) // 2]
            return negative_candidates[median_idx]

        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")


class BatchTripletMiner:
    """
    Mines triplets from a batch of embeddings during training

    Uses online hard mining within a batch for efficiency.
    """

    def __init__(
        self,
        margin: float = 0.1,
        mining_strategy: str = "hard"
    ):
        """
        Initialize batch triplet miner

        Args:
            margin: Triplet loss margin
            mining_strategy: Mining strategy (hard, semi-hard, all)
        """
        self.margin = margin
        self.mining_strategy = mining_strategy

    def mine_batch_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine triplets from batch

        Args:
            embeddings: (batch_size, embedding_dim) embeddings
            labels: (batch_size,) labels (keyframe IDs or cluster IDs)

        Returns:
            anchors: (n_triplets, embedding_dim)
            positives: (n_triplets, embedding_dim)
            negatives: (n_triplets, embedding_dim)
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)

        # Create label masks
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_not_equal = ~label_equal

        # For each anchor
        anchors = []
        positives = []
        negatives = []

        for i in range(batch_size):
            # Find positives (same label, excluding self)
            positive_mask = label_equal[i].clone()
            positive_mask[i] = False

            if not positive_mask.any():
                continue  # No valid positives

            # Find negatives (different label)
            negative_mask = label_not_equal[i]

            if not negative_mask.any():
                continue  # No valid negatives

            # Select positive
            if self.mining_strategy == "hard":
                # Hardest positive = farthest
                positive_distances = distances[i].clone()
                positive_distances[~positive_mask] = -1
                positive_idx = positive_distances.argmax()
            else:
                # Random positive
                positive_indices = positive_mask.nonzero(as_tuple=True)[0]
                positive_idx = positive_indices[torch.randint(len(positive_indices), (1,))]

            # Select negative
            if self.mining_strategy == "hard":
                # Hardest negative = closest
                negative_distances = distances[i].clone()
                negative_distances[~negative_mask] = float('inf')
                negative_idx = negative_distances.argmin()
            elif self.mining_strategy == "semi-hard":
                # Semi-hard: d(a,n) > d(a,p) but < d(a,p) + margin
                positive_dist = distances[i, positive_idx]
                negative_distances = distances[i].clone()
                negative_distances[~negative_mask] = float('inf')

                # Find semi-hard negatives
                semi_hard_mask = (
                    (negative_distances > positive_dist) &
                    (negative_distances < positive_dist + self.margin)
                )

                if semi_hard_mask.any():
                    semi_hard_distances = negative_distances.clone()
                    semi_hard_distances[~semi_hard_mask] = float('inf')
                    negative_idx = semi_hard_distances.argmin()
                else:
                    # Fall back to hardest
                    negative_idx = negative_distances.argmin()
            else:
                # Random negative
                negative_indices = negative_mask.nonzero(as_tuple=True)[0]
                negative_idx = negative_indices[torch.randint(len(negative_indices), (1,))]

            # Add triplet
            anchors.append(embeddings[i])
            positives.append(embeddings[positive_idx])
            negatives.append(embeddings[negative_idx])

        if len(anchors) == 0:
            # No valid triplets
            return (
                torch.zeros((0, embeddings.shape[1]), device=device),
                torch.zeros((0, embeddings.shape[1]), device=device),
                torch.zeros((0, embeddings.shape[1]), device=device)
            )

        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives)
        )

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise L2 distances

        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            (batch_size, batch_size) distance matrix
        """
        # Efficient pairwise distance computation
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

        dot_product = torch.mm(embeddings, embeddings.t())
        squared_norms = torch.diag(dot_product).unsqueeze(0)

        distances = squared_norms + squared_norms.t() - 2 * dot_product
        distances = torch.clamp(distances, min=0.0)  # Numerical stability

        return torch.sqrt(distances)


def create_triplet_miner(
    positive_distance_max: float = 5.0,
    positive_temporal_min: int = 30,
    negative_distance_min: float = 10.0,
    negative_distance_max: float = 50.0,
    mining_strategy: str = "hard"
) -> TripletMiner:
    """
    Factory function to create triplet miner

    Args:
        positive_distance_max: Max distance for positives (meters)
        positive_temporal_min: Min temporal gap for positives (frames)
        negative_distance_min: Min distance for negatives (meters)
        negative_distance_max: Max distance for negatives (meters)
        mining_strategy: Mining strategy

    Returns:
        TripletMiner instance
    """
    return TripletMiner(
        positive_distance_max=positive_distance_max,
        positive_temporal_min=positive_temporal_min,
        negative_distance_min=negative_distance_min,
        negative_distance_max=negative_distance_max,
        mining_strategy=mining_strategy
    )

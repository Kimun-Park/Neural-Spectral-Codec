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
import logging
import time
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from data.pose_utils import euclidean_distance
from retrieval.wasserstein import wasserstein_distance_1d_numpy


class TripletMiner:
    """
    Mines triplets (anchor, positive, negative) for training.

    Loop Closure Mining Strategy:
    - Positive: Same location (< 5m) but different time (>= skip_frames apart)
    - Negative: Different location (10-50m) AND different time (>= skip_frames apart)

    This ensures the model learns to recognize places across time,
    not just temporal similarity.
    """

    def __init__(
        self,
        positive_distance_max: float = 5.0,
        positive_temporal_min: int = 30,
        negative_distance_min: float = 10.0,
        negative_distance_max: float = 50.0,
        negative_temporal_min: int = 30,
        mining_strategy: str = "hard"
    ):
        """
        Initialize triplet miner for loop closure learning.

        Args:
            positive_distance_max: Max distance for positive pairs (meters)
            positive_temporal_min: Min temporal gap for positives (keyframes)
            negative_distance_min: Min distance for negatives (meters)
            negative_distance_max: Max distance for negatives (meters)
            negative_temporal_min: Min temporal gap for negatives (keyframes)
            mining_strategy: Mining strategy (hard, semi-hard, random)
        """
        self.positive_distance_max = positive_distance_max
        self.positive_temporal_min = positive_temporal_min
        self.negative_distance_min = negative_distance_min
        self.negative_distance_max = negative_distance_max
        self.negative_temporal_min = negative_temporal_min
        self.mining_strategy = mining_strategy

    def mine_triplets(
        self,
        descriptors: np.ndarray,
        poses: np.ndarray,
        n_triplets_per_anchor: int = 1,
        sequence_ids: np.ndarray = None
    ) -> List[Tuple[int, int, int]]:
        """
        Mine triplets from keyframe data (per-sequence to avoid cross-sequence pairs)

        Args:
            descriptors: (n_keyframes, n_bins) spectral histograms
            poses: (n_keyframes, 4, 4) SE(3) poses
            n_triplets_per_anchor: Number of triplets per anchor
            sequence_ids: (n_keyframes,) sequence ID for each keyframe (optional)

        Returns:
            List of (anchor_idx, positive_idx, negative_idx) tuples
        """
        n_keyframes = len(descriptors)
        triplets = []

        # If sequence_ids provided, mine per sequence (much faster)
        if sequence_ids is not None:
            unique_seqs = np.unique(sequence_ids)
            logging.info(f"Mining triplets per sequence ({len(unique_seqs)} sequences)...")

            for seq_idx, seq_id in enumerate(unique_seqs):
                seq_start = time.perf_counter()
                seq_mask = sequence_ids == seq_id
                seq_indices = np.where(seq_mask)[0]

                if len(seq_indices) < 3:
                    continue

                # Mine within this sequence
                seq_triplets = self._mine_sequence_triplets(
                    seq_indices, descriptors, poses, n_triplets_per_anchor
                )
                triplets.extend(seq_triplets)
                seq_time = time.perf_counter() - seq_start

                logging.info(
                    f"  Seq {seq_idx+1}/{len(unique_seqs)} (id={seq_id}): "
                    f"{len(seq_indices):,} keyframes -> {len(seq_triplets):,} triplets "
                    f"({seq_time:.1f}s, {len(seq_triplets)/seq_time:.0f}/s)"
                )

            return triplets

        # Original O(n²) approach if no sequence_ids
        for anchor_idx in range(n_keyframes):
            positive_candidates = self._find_positive_candidates(
                anchor_idx, poses, n_keyframes
            )

            if len(positive_candidates) == 0:
                continue

            negative_candidates = self._find_negative_candidates(
                anchor_idx, poses, n_keyframes
            )

            if len(negative_candidates) == 0:
                continue

            for _ in range(n_triplets_per_anchor):
                positive_idx = np.random.choice(positive_candidates)
                negative_idx = self._select_hard_negative(
                    anchor_idx, negative_candidates, descriptors
                )
                triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def _mine_sequence_triplets(
        self,
        seq_indices: np.ndarray,
        descriptors: np.ndarray,
        poses: np.ndarray,
        n_triplets_per_anchor: int
    ) -> List[Tuple[int, int, int]]:
        """
        Mine triplets within a single sequence using KD-Tree (O(n log n) instead of O(n²)).

        Loop Closure Mining:
        - Positive: distance < 5m AND temporal_gap >= 30 (same place, different time)
        - Negative: 10m < distance < 50m AND temporal_gap >= 30 (different place, different time)

        This ensures the model learns place recognition across time,
        excluding temporal neighbors from both positives and negatives.
        """
        triplets = []
        n_seq = len(seq_indices)

        # Extract positions from poses (translation component)
        seq_positions = np.array([
            poses[idx][:3, 3] for idx in seq_indices
        ])  # (n_seq, 3)

        # Build KD-Tree for fast spatial queries - O(n log n)
        tree = cKDTree(seq_positions)

        # Statistics
        n_anchors_with_positives = 0
        n_anchors_with_negatives = 0

        for local_anchor in range(n_seq):
            anchor_idx = seq_indices[local_anchor]
            anchor_pos = seq_positions[local_anchor]

            # Query positives: within positive_distance_max - O(log n)
            positive_local_indices = tree.query_ball_point(
                anchor_pos, r=self.positive_distance_max
            )

            # Query negatives: within negative_distance_max - O(log n)
            neg_outer_indices = tree.query_ball_point(
                anchor_pos, r=self.negative_distance_max
            )
            neg_inner_indices = set(tree.query_ball_point(
                anchor_pos, r=self.negative_distance_min
            ))

            # Filter positives by temporal gap (loop closure constraint)
            positive_candidates = []
            for local_other in positive_local_indices:
                if local_other == local_anchor:
                    continue
                temporal_gap = abs(local_other - local_anchor)
                if temporal_gap >= self.positive_temporal_min:
                    positive_candidates.append(seq_indices[local_other])

            # Filter negatives:
            # 1. In outer ring (between min and max distance)
            # 2. NOT temporal neighbors (temporal_gap >= negative_temporal_min)
            negative_candidates = []
            for local_other in neg_outer_indices:
                if local_other == local_anchor:
                    continue
                # Spatial constraint: in 10-50m ring
                if local_other in neg_inner_indices:
                    continue
                # Temporal constraint: not a temporal neighbor
                temporal_gap = abs(local_other - local_anchor)
                if temporal_gap >= self.negative_temporal_min:
                    negative_candidates.append(seq_indices[local_other])

            if len(positive_candidates) > 0:
                n_anchors_with_positives += 1
            if len(negative_candidates) > 0:
                n_anchors_with_negatives += 1

            if len(positive_candidates) == 0 or len(negative_candidates) == 0:
                continue

            for _ in range(n_triplets_per_anchor):
                positive_idx = np.random.choice(positive_candidates)
                negative_idx = self._select_hard_negative(
                    anchor_idx, negative_candidates, descriptors
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
        Find negative candidates for anchor (loop closure style).

        Criteria:
        - 10m < distance < 50m (different location)
        - temporal_gap >= negative_temporal_min (not a temporal neighbor)

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

            # Check temporal gap (exclude temporal neighbors)
            temporal_gap = abs(i - anchor_idx)
            if temporal_gap < self.negative_temporal_min:
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
    negative_temporal_min: int = 30,
    mining_strategy: str = "hard"
) -> TripletMiner:
    """
    Factory function to create triplet miner for loop closure learning.

    Args:
        positive_distance_max: Max distance for positives (meters)
        positive_temporal_min: Min temporal gap for positives (keyframes)
        negative_distance_min: Min distance for negatives (meters)
        negative_distance_max: Max distance for negatives (meters)
        negative_temporal_min: Min temporal gap for negatives (keyframes)
        mining_strategy: Mining strategy (hard, semi-hard, random)

    Returns:
        TripletMiner instance configured for loop closure learning
    """
    return TripletMiner(
        positive_distance_max=positive_distance_max,
        positive_temporal_min=positive_temporal_min,
        negative_distance_min=negative_distance_min,
        negative_distance_max=negative_distance_max,
        negative_temporal_min=negative_temporal_min,
        mining_strategy=mining_strategy
    )

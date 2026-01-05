"""
Graph Manager for Temporal Keyframe Graph

Manages PyTorch Geometric graph structure with:
- Temporal edges (M=5 nearest neighbors)
- Sliding window (max 1000 active nodes)
- Local updates (3-hop neighborhoods)
- Efficient graph lifecycle management
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Set
from keyframe.selector import Keyframe


class TemporalGraphManager:
    """
    Manages temporal graph of keyframes for GNN processing

    Creates and maintains a PyTorch Geometric graph where:
    - Nodes are keyframes with spectral histogram features
    - Edges connect temporally adjacent keyframes (M=5 neighbors)
    - Sliding window maintains max 1000 active nodes
    - Old nodes are frozen (embeddings cached, removed from active computation)
    """

    def __init__(
        self,
        temporal_neighbors: int = 5,
        max_active_nodes: int = 1000,
        feature_dim: int = 50,
        device: str = 'cpu'
    ):
        """
        Initialize graph manager

        Args:
            temporal_neighbors: Number of temporal neighbors (M=5)
            max_active_nodes: Maximum active nodes in sliding window
            feature_dim: Feature dimension (matches histogram bins)
            device: Device for PyTorch tensors
        """
        self.temporal_neighbors = temporal_neighbors
        self.max_active_nodes = max_active_nodes
        self.feature_dim = feature_dim
        self.device = device

        # Active graph
        self.graph: Optional[Data] = None
        self.keyframes: List[Keyframe] = []

        # Frozen nodes (beyond sliding window)
        self.frozen_keyframes: List[Keyframe] = []
        self.frozen_embeddings: Optional[torch.Tensor] = None

        # Node index mapping
        self.keyframe_id_to_node_idx = {}

    def reset(self):
        """Reset graph state"""
        self.graph = None
        self.keyframes.clear()
        self.frozen_keyframes.clear()
        self.frozen_embeddings = None
        self.keyframe_id_to_node_idx.clear()

    def add_keyframe(self, keyframe: Keyframe) -> int:
        """
        Add new keyframe to graph

        Args:
            keyframe: Keyframe to add (must have descriptor set)

        Returns:
            Node index in active graph
        """
        if keyframe.descriptor is None:
            raise ValueError("Keyframe must have descriptor computed before adding to graph")

        # Add to keyframes list
        self.keyframes.append(keyframe)
        node_idx = len(self.keyframes) - 1

        # Update mapping
        self.keyframe_id_to_node_idx[keyframe.keyframe_id] = node_idx

        # Rebuild graph
        self._rebuild_graph()

        # Check sliding window constraint
        if len(self.keyframes) > self.max_active_nodes:
            self._freeze_oldest_node()

        return node_idx

    def _rebuild_graph(self):
        """
        Rebuild PyTorch Geometric graph from current keyframes
        """
        n_nodes = len(self.keyframes)

        if n_nodes == 0:
            self.graph = None
            return

        # Extract features (descriptors)
        features = torch.stack([
            torch.from_numpy(kf.descriptor).float()
            for kf in self.keyframes
        ], dim=0).to(self.device)  # (n_nodes, feature_dim)

        # Build temporal edges (M nearest neighbors)
        edge_index = self._build_temporal_edges(n_nodes)

        # Create PyG Data object
        self.graph = Data(
            x=features,
            edge_index=edge_index,
            num_nodes=n_nodes
        ).to(self.device)

    def _build_temporal_edges(self, n_nodes: int) -> torch.Tensor:
        """
        Build temporal edge connections (M=5 nearest neighbors in time)

        Args:
            n_nodes: Number of nodes

        Returns:
            (2, num_edges) edge index tensor
        """
        edges = []

        for i in range(n_nodes):
            # Connect to M/2 past neighbors and M/2 future neighbors
            half_window = self.temporal_neighbors // 2

            for offset in range(-half_window, half_window + 1):
                if offset == 0:
                    continue

                neighbor_idx = i + offset

                # Check bounds
                if 0 <= neighbor_idx < n_nodes:
                    # Bidirectional edge
                    edges.append([i, neighbor_idx])

        if len(edges) == 0:
            # No edges (single node)
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()

        return edge_index

    def _freeze_oldest_node(self):
        """
        Freeze oldest node and move to frozen storage

        Freezing means:
        - Remove from active graph
        - Cache embedding (if computed)
        - Keep descriptor for retrieval
        """
        if len(self.keyframes) == 0:
            return

        # Remove oldest keyframe
        oldest_kf = self.keyframes.pop(0)
        self.frozen_keyframes.append(oldest_kf)

        # Update node index mapping
        del self.keyframe_id_to_node_idx[oldest_kf.keyframe_id]

        # Shift all indices down by 1
        for kf_id in self.keyframe_id_to_node_idx:
            self.keyframe_id_to_node_idx[kf_id] -= 1

        # Cache embedding if available
        if oldest_kf.embedding is not None:
            embedding = torch.from_numpy(oldest_kf.embedding).float().to(self.device)

            if self.frozen_embeddings is None:
                self.frozen_embeddings = embedding.unsqueeze(0)
            else:
                self.frozen_embeddings = torch.cat([
                    self.frozen_embeddings,
                    embedding.unsqueeze(0)
                ], dim=0)

        # Rebuild graph without oldest node
        self._rebuild_graph()

    def get_graph(self) -> Optional[Data]:
        """Get current active graph"""
        return self.graph

    def get_node_index(self, keyframe_id: int) -> Optional[int]:
        """
        Get node index for keyframe ID in active graph

        Args:
            keyframe_id: Keyframe ID

        Returns:
            Node index or None if not in active graph
        """
        return self.keyframe_id_to_node_idx.get(keyframe_id, None)

    def get_k_hop_neighbors(self, node_idx: int, k: int) -> Set[int]:
        """
        Get k-hop neighborhood of a node

        Args:
            node_idx: Node index
            k: Number of hops

        Returns:
            Set of node indices in k-hop neighborhood
        """
        if self.graph is None or k <= 0:
            return {node_idx}

        # BFS to find k-hop neighbors
        neighbors = {node_idx}
        current_layer = {node_idx}

        edge_index = self.graph.edge_index.cpu().numpy()

        for _ in range(k):
            next_layer = set()

            for node in current_layer:
                # Find neighbors
                outgoing = edge_index[1, edge_index[0] == node]
                next_layer.update(outgoing.tolist())

            neighbors.update(next_layer)
            current_layer = next_layer

            if len(current_layer) == 0:
                break

        return neighbors

    def get_local_subgraph(self, node_idx: int, k_hops: int = 3) -> Tuple[Data, dict]:
        """
        Extract k-hop local subgraph around a node

        Args:
            node_idx: Center node index
            k_hops: Number of hops (default: 3)

        Returns:
            subgraph: Local subgraph as PyG Data
            mapping: Mapping from original to subgraph indices
        """
        if self.graph is None:
            raise ValueError("Graph is empty")

        # Get k-hop neighbors
        neighbor_indices = sorted(list(self.get_k_hop_neighbors(node_idx, k_hops)))

        # Create mapping
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(neighbor_indices)}

        # Extract subgraph features
        subgraph_features = self.graph.x[neighbor_indices]

        # Extract subgraph edges
        edge_index = self.graph.edge_index.cpu().numpy()
        subgraph_edges = []

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]

            if src in neighbor_indices and dst in neighbor_indices:
                # Remap to subgraph indices
                new_src = mapping[src]
                new_dst = mapping[dst]
                subgraph_edges.append([new_src, new_dst])

        if len(subgraph_edges) > 0:
            subgraph_edge_index = torch.tensor(
                subgraph_edges,
                dtype=torch.long,
                device=self.device
            ).t()
        else:
            subgraph_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        # Create subgraph
        subgraph = Data(
            x=subgraph_features,
            edge_index=subgraph_edge_index,
            num_nodes=len(neighbor_indices)
        ).to(self.device)

        return subgraph, mapping

    def update_embeddings(self, embeddings: torch.Tensor):
        """
        Update keyframe embeddings from GNN output

        Args:
            embeddings: (n_nodes, embedding_dim) GNN embeddings
        """
        if len(embeddings) != len(self.keyframes):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) != keyframe count ({len(self.keyframes)})"
            )

        # Update keyframe embeddings
        embeddings_np = embeddings.detach().cpu().numpy()

        for i, kf in enumerate(self.keyframes):
            kf.embedding = embeddings_np[i]

    def get_all_keyframes(self) -> List[Keyframe]:
        """Get all keyframes (active + frozen)"""
        return self.frozen_keyframes + self.keyframes

    def get_all_descriptors(self) -> np.ndarray:
        """
        Get all descriptors (active + frozen)

        Returns:
            (total_keyframes, feature_dim) array
        """
        all_kfs = self.get_all_keyframes()

        descriptors = np.array([kf.descriptor for kf in all_kfs])

        return descriptors

    def get_all_embeddings(self) -> Optional[np.ndarray]:
        """
        Get all embeddings (active + frozen)

        Returns:
            (total_keyframes, embedding_dim) array or None
        """
        all_kfs = self.get_all_keyframes()

        if all_kfs[0].embedding is None:
            return None

        embeddings = np.array([kf.embedding for kf in all_kfs])

        return embeddings

    def get_statistics(self) -> dict:
        """Get graph statistics"""
        return {
            'num_active_nodes': len(self.keyframes),
            'num_frozen_nodes': len(self.frozen_keyframes),
            'total_nodes': len(self.keyframes) + len(self.frozen_keyframes),
            'num_edges': self.graph.edge_index.shape[1] if self.graph is not None else 0,
            'avg_degree': (
                self.graph.edge_index.shape[1] / len(self.keyframes)
                if self.graph is not None and len(self.keyframes) > 0
                else 0.0
            )
        }


def build_graph_from_keyframes(
    keyframes: List[Keyframe],
    temporal_neighbors: int = 5,
    device: str = 'cpu'
) -> Data:
    """
    Build PyTorch Geometric graph from keyframe list

    Args:
        keyframes: List of keyframes with descriptors
        temporal_neighbors: Number of temporal neighbors
        device: Device for tensors

    Returns:
        PyG Data object
    """
    manager = TemporalGraphManager(
        temporal_neighbors=temporal_neighbors,
        max_active_nodes=len(keyframes),  # No freezing for offline construction
        device=device
    )

    for kf in keyframes:
        manager.add_keyframe(kf)

    return manager.get_graph()

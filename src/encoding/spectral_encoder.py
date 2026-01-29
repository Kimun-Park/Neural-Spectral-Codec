"""
Spectral Histogram Encoder - Algorithm 1

Implements the core Neural Spectral Codec algorithm:
1. Convert point cloud to panoramic range image (64 × 360)
2. Apply ring-wise 1D FFT along azimuth for rotation invariance
3. Aggregate FFT magnitudes into per-elevation histograms via exponential binning
4. Normalize and prepare for quantization

Key features:
- Rotation invariance via magnitude spectrum (phase discarded)
- Per-elevation histogram preserves height information (16 × 50 = 800D)
- Adaptive exponential frequency binning (learnable α parameter)
- Numerical stability via proper normalization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from encoding.range_image import RangeImageProjector, interpolate_range_image


class SpectralEncoder(nn.Module):
    """
    Neural Spectral Histogram Encoder

    Compresses LiDAR point clouds to per-elevation spectral histograms.
    Output dimension: target_elevation_bins × n_bins (e.g., 16 × 50 = 800D)

    This preserves height-specific spectral information for better
    cross-sensor generalization.
    """

    def __init__(
        self,
        n_elevation: int = 64,
        n_azimuth: int = 360,
        n_bins: int = 50,
        alpha: float = 2.0,
        learnable_alpha: bool = True,
        epsilon: float = 1e-8,
        target_elevation_bins: int = 16,
        interpolate_empty: bool = True,
        elevation_range: tuple = (-24.8, 2.0),
        device: str = 'cpu'
    ):
        """
        Initialize spectral encoder

        Args:
            n_elevation: Number of elevation rings (64 for HDL-64E)
            n_azimuth: Number of azimuth bins (360 for 1-degree resolution)
            n_bins: Number of histogram bins (50 for target compression)
            alpha: Exponential warping parameter for frequency binning
            learnable_alpha: If True, α is learned during training
            epsilon: Small constant for numerical stability
            target_elevation_bins: Target elevation bins for sensor-agnostic binning (16 for compatibility)
            interpolate_empty: If True, interpolate empty pixels before FFT (critical for sensor invariance)
            elevation_range: (min, max) elevation angles in degrees (sensor-specific)
            device: Device for tensor operations
        """
        super().__init__()

        self.n_elevation = n_elevation
        self.n_azimuth = n_azimuth
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.target_elevation_bins = target_elevation_bins
        self.interpolate_empty = interpolate_empty
        self._device = device

        # Learnable alpha parameter
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        # Range image projector with sensor-specific elevation range
        self.projector = RangeImageProjector(
            n_elevation=n_elevation,
            n_azimuth=n_azimuth,
            elevation_range=elevation_range
        )

        # Precompute number of FFT frequencies
        # Real FFT outputs (n_azimuth // 2 + 1) frequencies
        self.n_freqs = n_azimuth // 2 + 1  # 181 for 360 azimuth bins

        # Output dimension: per-elevation histograms
        self.output_dim = target_elevation_bins * n_bins  # 16 * 50 = 800

    def _compute_bin_edges(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive exponential frequency bin edges

        Maps [0, n_freqs] to [0, n_bins] using exponential warping:
        bin_edge[i] = (exp(α * i / n_bins) - 1) / (exp(α) - 1) * n_freqs

        Args:
            alpha: Warping parameter

        Returns:
            (n_bins + 1,) bin edges in frequency space
        """
        # Linear spacing in warped space
        t = torch.linspace(0, 1, self.n_bins + 1, device=alpha.device)

        # Exponential warping
        # f(t) = (exp(α*t) - 1) / (exp(α) - 1)
        bin_edges = (torch.exp(alpha * t) - 1) / (torch.exp(alpha) - 1 + self.epsilon)

        # Scale to frequency range [0, n_freqs]
        bin_edges = bin_edges * self.n_freqs

        return bin_edges

    def _bin_fft_magnitudes(
        self,
        fft_magnitudes: torch.Tensor,
        bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Bin FFT magnitudes into per-elevation histograms using adaptive edges

        Args:
            fft_magnitudes: (n_elevation, n_freqs) FFT magnitude spectrum
            bin_edges: (n_bins + 1,) bin edges in frequency space

        Returns:
            (n_elevation * n_bins,) flattened per-elevation histograms
        """
        n_elevation = fft_magnitudes.shape[0]

        # Frequency indices: (n_freqs,)
        freq_indices = torch.arange(
            self.n_freqs,
            dtype=torch.float32,
            device=fft_magnitudes.device
        )

        # Assign each frequency to a bin using searchsorted (vectorized)
        # bin_assignments[j] = i means frequency j belongs to bin i
        bin_assignments = torch.searchsorted(bin_edges, freq_indices, right=True) - 1
        bin_assignments = torch.clamp(bin_assignments, 0, self.n_bins - 1)

        # Create per-elevation histograms
        # fft_magnitudes: (n_elevation, n_freqs)
        histograms = torch.zeros(n_elevation, self.n_bins, device=fft_magnitudes.device)

        # Scatter add for each elevation
        for elev_idx in range(n_elevation):
            histograms[elev_idx].scatter_add_(
                0, bin_assignments.long(), fft_magnitudes[elev_idx]
            )

        # Flatten to (n_elevation * n_bins,)
        return histograms.flatten()

    def encode_range_image(self, range_image: torch.Tensor) -> torch.Tensor:
        """
        Encode range image to per-elevation spectral histogram

        Args:
            range_image: (n_elevation, n_azimuth) range values

        Returns:
            (target_elevation_bins * n_bins,) globally normalized per-elevation histogram
        """
        # Sensor-agnostic elevation binning
        if range_image.shape[0] != self.target_elevation_bins:
            # Adaptive average pooling to normalize elevation dimension
            range_image = torch.nn.functional.adaptive_avg_pool2d(
                range_image.unsqueeze(0).unsqueeze(0),
                (self.target_elevation_bins, range_image.shape[1])
            ).squeeze()

        # Apply ring-wise 1D FFT along azimuth dimension
        # FFT along last dimension (azimuth)
        fft_output = torch.fft.rfft(range_image, dim=1, norm='ortho')

        # Take magnitude (discard phase for rotation invariance)
        fft_magnitudes = torch.abs(fft_output)  # (target_elevation_bins, n_freqs)

        # Normalize by sqrt(n_azimuth) to maintain scale
        fft_magnitudes = fft_magnitudes * np.sqrt(self.n_azimuth)

        # Compute adaptive bin edges
        bin_edges = self._compute_bin_edges(self.alpha)

        # Bin FFT magnitudes into per-elevation histograms
        histogram = self._bin_fft_magnitudes(fft_magnitudes, bin_edges)
        # histogram shape: (target_elevation_bins * n_bins,)

        # Global normalization: entire histogram sums to 1
        # This preserves relative point density between elevations
        histogram_sum = histogram.sum()
        if histogram_sum > self.epsilon:
            histogram = histogram / (histogram_sum + self.epsilon)
        else:
            # Empty histogram - uniform distribution
            histogram = torch.ones_like(histogram) / histogram.numel()

        return histogram

    def encode_points(self, points: np.ndarray) -> torch.Tensor:
        """
        Encode point cloud to spectral histogram

        Args:
            points: (N, 3) or (N, 4) numpy array [x, y, z] or [x, y, z, intensity]

        Returns:
            (n_bins,) normalized spectral histogram
        """
        # Project to range image
        range_image, _ = self.projector.project(points, keep_intensity=False)

        # Interpolate empty pixels for sensor-invariant FFT
        if self.interpolate_empty:
            range_image = interpolate_range_image(range_image, method='linear')

        # Convert to torch tensor and move to same device as model
        range_image_tensor = torch.from_numpy(range_image).float().to(self.alpha.device)

        # Encode
        histogram = self.encode_range_image(range_image_tensor)

        return histogram

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (for use in neural network training)

        Args:
            x: (batch, n_elevation, n_azimuth) batch of range images

        Returns:
            (batch, output_dim) batch of per-elevation spectral histograms
            where output_dim = target_elevation_bins * n_bins (e.g., 800)
        """
        batch_size = x.shape[0]
        histograms = []

        for i in range(batch_size):
            histogram = self.encode_range_image(x[i])
            histograms.append(histogram)

        return torch.stack(histograms, dim=0)

    def encode_batch(self, range_images: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of range images

        Args:
            range_images: (batch, n_elevation, n_azimuth) range images

        Returns:
            (batch, n_bins) spectral histograms
        """
        return self.forward(range_images)


class SpectralEncoderNumpy:
    """
    Numpy-only version of SpectralEncoder for inference without PyTorch

    Useful for deployment scenarios where PyTorch is not available.
    """

    def __init__(
        self,
        n_elevation: int = 64,
        n_azimuth: int = 360,
        n_bins: int = 50,
        alpha: float = 2.0,
        epsilon: float = 1e-8
    ):
        """
        Initialize numpy spectral encoder

        Args:
            n_elevation: Number of elevation rings
            n_azimuth: Number of azimuth bins
            n_bins: Number of histogram bins
            alpha: Exponential warping parameter
            epsilon: Numerical stability constant
        """
        self.n_elevation = n_elevation
        self.n_azimuth = n_azimuth
        self.n_bins = n_bins
        self.alpha = alpha
        self.epsilon = epsilon

        self.projector = RangeImageProjector(
            n_elevation=n_elevation,
            n_azimuth=n_azimuth
        )

        self.n_freqs = n_azimuth // 2 + 1

    def _compute_bin_edges(self) -> np.ndarray:
        """Compute adaptive exponential frequency bin edges"""
        t = np.linspace(0, 1, self.n_bins + 1)
        bin_edges = (np.exp(self.alpha * t) - 1) / (np.exp(self.alpha) - 1 + self.epsilon)
        bin_edges = bin_edges * self.n_freqs
        return bin_edges

    def encode_range_image(self, range_image: np.ndarray) -> np.ndarray:
        """
        Encode range image to spectral histogram

        Args:
            range_image: (n_elevation, n_azimuth) range values

        Returns:
            (n_bins,) normalized spectral histogram
        """
        # Apply ring-wise 1D FFT
        fft_output = np.fft.rfft(range_image, axis=1, norm='ortho')

        # Take magnitude
        fft_magnitudes = np.abs(fft_output)

        # Normalize
        fft_magnitudes = fft_magnitudes * np.sqrt(self.n_azimuth)

        # Compute bin edges
        bin_edges = self._compute_bin_edges()

        # Initialize histogram
        histogram = np.zeros(self.n_bins)

        # Bin magnitudes
        freq_indices = np.arange(self.n_freqs)

        for i in range(self.n_bins):
            mask = (freq_indices >= bin_edges[i]) & (freq_indices < bin_edges[i + 1])
            if mask.any():
                histogram[i] = fft_magnitudes[:, mask].sum()

        # Normalize
        histogram_sum = histogram.sum()
        if histogram_sum > self.epsilon:
            histogram = histogram / (histogram_sum + self.epsilon)
        else:
            histogram = np.ones(self.n_bins) / self.n_bins

        return histogram

    def encode_points(self, points: np.ndarray) -> np.ndarray:
        """
        Encode point cloud to spectral histogram

        Args:
            points: (N, 3) or (N, 4) point cloud

        Returns:
            (n_bins,) spectral histogram
        """
        range_image, _ = self.projector.project(points, keep_intensity=False)
        return self.encode_range_image(range_image)


def test_rotation_invariance(
    encoder: SpectralEncoder,
    points: np.ndarray,
    n_rotations: int = 8
) -> float:
    """
    Test rotation invariance of spectral encoder

    Args:
        encoder: SpectralEncoder instance
        points: (N, 3) point cloud
        n_rotations: Number of rotations to test

    Returns:
        Maximum histogram difference across rotations
    """
    from data.pose_utils import transform_points

    histograms = []

    for i in range(n_rotations):
        # Rotate around z-axis
        angle = 2 * np.pi * i / n_rotations

        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Rotate points
        rotated_points = transform_points(points, R)

        # Encode
        histogram = encoder.encode_points(rotated_points)
        histograms.append(histogram.detach().numpy())

    # Compute maximum difference
    histograms = np.array(histograms)
    max_diff = 0.0

    for i in range(n_rotations):
        for j in range(i + 1, n_rotations):
            diff = np.abs(histograms[i] - histograms[j]).max()
            max_diff = max(max_diff, diff)

    return max_diff

"""
Spectral Histogram Encoder - Algorithm 1

Implements the core Neural Spectral Codec algorithm:
1. Convert point cloud to panoramic range image (64 × 360)
2. Apply ring-wise 1D FFT along azimuth for rotation invariance
3. Aggregate FFT magnitudes into 50-bin histogram via exponential binning
4. Normalize and prepare for quantization

Key features:
- Rotation invariance via magnitude spectrum (phase discarded)
- Adaptive exponential frequency binning (learnable α parameter)
- Numerical stability via proper normalization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from encoding.range_image import RangeImageProjector


class SpectralEncoder(nn.Module):
    """
    Neural Spectral Histogram Encoder

    Compresses LiDAR point clouds to 50-dimensional spectral histograms.
    """

    def __init__(
        self,
        n_elevation: int = 64,
        n_azimuth: int = 360,
        n_bins: int = 50,
        alpha: float = 2.0,
        learnable_alpha: bool = True,
        epsilon: float = 1e-8
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
        """
        super().__init__()

        self.n_elevation = n_elevation
        self.n_azimuth = n_azimuth
        self.n_bins = n_bins
        self.epsilon = epsilon

        # Learnable alpha parameter
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        # Range image projector
        self.projector = RangeImageProjector(
            n_elevation=n_elevation,
            n_azimuth=n_azimuth
        )

        # Precompute number of FFT frequencies
        # Real FFT outputs (n_azimuth // 2 + 1) frequencies
        self.n_freqs = n_azimuth // 2 + 1  # 181 for 360 azimuth bins

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
        Bin FFT magnitudes into histogram using adaptive edges

        Args:
            fft_magnitudes: (n_elevation, n_freqs) FFT magnitude spectrum
            bin_edges: (n_bins + 1,) bin edges in frequency space

        Returns:
            (n_bins,) histogram of aggregated magnitudes
        """
        # Frequency indices
        freq_indices = torch.arange(
            self.n_freqs,
            dtype=torch.float32,
            device=fft_magnitudes.device
        )

        # Initialize histogram
        histogram = torch.zeros(self.n_bins, device=fft_magnitudes.device)

        # Aggregate magnitudes into bins
        for i in range(self.n_bins):
            # Find frequencies in this bin
            mask = (freq_indices >= bin_edges[i]) & (freq_indices < bin_edges[i + 1])

            if mask.any():
                # Sum magnitudes across all elevation rings for frequencies in this bin
                histogram[i] = fft_magnitudes[:, mask].sum()

        return histogram

    def encode_range_image(self, range_image: torch.Tensor) -> torch.Tensor:
        """
        Encode range image to spectral histogram

        Args:
            range_image: (n_elevation, n_azimuth) range values

        Returns:
            (n_bins,) normalized spectral histogram
        """
        # Apply ring-wise 1D FFT along azimuth dimension
        # FFT along last dimension (azimuth)
        fft_output = torch.fft.rfft(range_image, dim=1, norm='ortho')

        # Take magnitude (discard phase for rotation invariance)
        fft_magnitudes = torch.abs(fft_output)  # (n_elevation, n_freqs)

        # Normalize by sqrt(n_azimuth) to maintain scale
        fft_magnitudes = fft_magnitudes * np.sqrt(self.n_azimuth)

        # Compute adaptive bin edges
        bin_edges = self._compute_bin_edges(self.alpha)

        # Bin FFT magnitudes into histogram
        histogram = self._bin_fft_magnitudes(fft_magnitudes, bin_edges)

        # Normalize histogram to sum to 1
        histogram_sum = histogram.sum()
        if histogram_sum > self.epsilon:
            histogram = histogram / (histogram_sum + self.epsilon)
        else:
            # Empty histogram - uniform distribution
            histogram = torch.ones(self.n_bins, device=histogram.device) / self.n_bins

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

        # Convert to torch tensor
        range_image_tensor = torch.from_numpy(range_image).float()

        # Encode
        histogram = self.encode_range_image(range_image_tensor)

        return histogram

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (for use in neural network training)

        Args:
            x: (batch, n_elevation, n_azimuth) batch of range images

        Returns:
            (batch, n_bins) batch of spectral histograms
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

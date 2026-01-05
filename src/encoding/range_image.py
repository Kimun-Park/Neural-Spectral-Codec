"""
Panoramic Range Image Projection for LiDAR Point Clouds

Converts 3D LiDAR point clouds to 2D panoramic range images
following the KITTI HDL-64E specifications:
- 64 elevation rings
- 360 azimuth bins (1 degree resolution)
"""

import numpy as np
from typing import Tuple, Optional


class RangeImageProjector:
    """
    Projects 3D point clouds to 2D panoramic range images

    Creates a 2D representation where:
    - Rows represent elevation rings (64 for HDL-64E)
    - Columns represent azimuth bins (360 for 1-degree resolution)
    - Values represent range (distance from sensor)
    """

    def __init__(
        self,
        n_elevation: int = 64,
        n_azimuth: int = 360,
        elevation_range: Tuple[float, float] = (-24.8, 2.0),
        max_range: float = 80.0,
        min_range: float = 1.0
    ):
        """
        Initialize range image projector

        Args:
            n_elevation: Number of elevation rings (64 for HDL-64E)
            n_azimuth: Number of azimuth bins (360 for 1-degree resolution)
            elevation_range: (min, max) elevation angles in degrees
            max_range: Maximum valid range in meters
            min_range: Minimum valid range in meters
        """
        self.n_elevation = n_elevation
        self.n_azimuth = n_azimuth
        self.max_range = max_range
        self.min_range = min_range

        # Convert elevation range to radians
        self.elevation_min = np.deg2rad(elevation_range[0])
        self.elevation_max = np.deg2rad(elevation_range[1])

    def project(
        self,
        points: np.ndarray,
        keep_intensity: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project 3D point cloud to 2D range image

        Args:
            points: (N, 3) or (N, 4) array of [x, y, z] or [x, y, z, intensity]
            keep_intensity: If True, also return intensity image

        Returns:
            range_image: (n_elevation, n_azimuth) range values
            intensity_image: (n_elevation, n_azimuth) intensity values (if keep_intensity)
        """
        # Extract coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Compute spherical coordinates
        # Range
        range_vals = np.sqrt(x**2 + y**2 + z**2)

        # Azimuth: angle in xy-plane from +x axis
        # Map from [-pi, pi] to [0, 2*pi] for easier binning
        azimuth = np.arctan2(y, x)  # [-pi, pi]
        azimuth = (azimuth + np.pi) % (2 * np.pi)  # [0, 2*pi]

        # Elevation: angle from xy-plane
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))

        # Filter points by range
        valid_mask = (range_vals >= self.min_range) & (range_vals <= self.max_range)
        range_vals = range_vals[valid_mask]
        azimuth = azimuth[valid_mask]
        elevation = elevation[valid_mask]

        if points.shape[1] == 4:
            intensity = points[:, 3][valid_mask]
        else:
            intensity = None

        # Bin into 2D grid
        # Elevation bins: map [-24.8°, 2.0°] to [0, 63]
        elev_norm = (elevation - self.elevation_min) / (self.elevation_max - self.elevation_min)
        elev_bins = np.clip(
            np.floor(elev_norm * self.n_elevation).astype(int),
            0,
            self.n_elevation - 1
        )

        # Azimuth bins: map [0, 2*pi] to [0, 359]
        azim_bins = np.clip(
            np.floor(azimuth / (2 * np.pi) * self.n_azimuth).astype(int),
            0,
            self.n_azimuth - 1
        )

        # Initialize images
        range_image = np.zeros((self.n_elevation, self.n_azimuth), dtype=np.float32)
        if keep_intensity and intensity is not None:
            intensity_image = np.zeros((self.n_elevation, self.n_azimuth), dtype=np.float32)
        else:
            intensity_image = None

        # Fill range image
        # For overlapping points, keep the closest (smallest range)
        range_image.fill(np.inf)

        for i in range(len(range_vals)):
            row = elev_bins[i]
            col = azim_bins[i]

            # Keep closest point
            if range_vals[i] < range_image[row, col]:
                range_image[row, col] = range_vals[i]

                if intensity_image is not None:
                    intensity_image[row, col] = intensity[i]

        # Replace inf with 0 for empty cells
        range_image[range_image == np.inf] = 0.0

        return range_image, intensity_image

    def unproject(
        self,
        range_image: np.ndarray,
        intensity_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert range image back to 3D point cloud

        Args:
            range_image: (n_elevation, n_azimuth) range values
            intensity_image: Optional (n_elevation, n_azimuth) intensity values

        Returns:
            points: (N, 3) or (N, 4) point cloud
        """
        # Get valid points (non-zero range)
        valid_mask = range_image > 0

        # Compute elevation and azimuth for each pixel
        elevation_bins = np.arange(self.n_elevation)
        azimuth_bins = np.arange(self.n_azimuth)

        # Create meshgrid
        elev_grid, azim_grid = np.meshgrid(elevation_bins, azimuth_bins, indexing='ij')

        # Convert bins to angles
        # Elevation: [0, 63] -> [-24.8°, 2.0°]
        elevation = (
            self.elevation_min +
            (elev_grid / self.n_elevation) * (self.elevation_max - self.elevation_min)
        )

        # Azimuth: [0, 359] -> [0, 2*pi]
        azimuth = (azim_grid / self.n_azimuth) * 2 * np.pi

        # Get ranges at valid locations
        ranges = range_image[valid_mask]
        elevations = elevation[valid_mask]
        azimuths = azimuth[valid_mask]

        # Convert to Cartesian
        x = ranges * np.cos(elevations) * np.cos(azimuths)
        y = ranges * np.cos(elevations) * np.sin(azimuths)
        z = ranges * np.sin(elevations)

        if intensity_image is not None:
            intensities = intensity_image[valid_mask]
            points = np.stack([x, y, z, intensities], axis=1)
        else:
            points = np.stack([x, y, z], axis=1)

        return points

    def visualize_range_image(self, range_image: np.ndarray) -> np.ndarray:
        """
        Convert range image to normalized visualization

        Args:
            range_image: (n_elevation, n_azimuth) range values

        Returns:
            (n_elevation, n_azimuth) normalized image in [0, 1]
        """
        # Normalize to [0, 1] based on max_range
        vis = np.clip(range_image / self.max_range, 0, 1)
        return vis


def project_to_range_image(
    points: np.ndarray,
    n_elevation: int = 64,
    n_azimuth: int = 360
) -> np.ndarray:
    """
    Convenience function to project points to range image

    Args:
        points: (N, 3) or (N, 4) point cloud
        n_elevation: Number of elevation rings
        n_azimuth: Number of azimuth bins

    Returns:
        (n_elevation, n_azimuth) range image
    """
    projector = RangeImageProjector(
        n_elevation=n_elevation,
        n_azimuth=n_azimuth
    )
    range_image, _ = projector.project(points, keep_intensity=False)
    return range_image


def compute_range_image_difference(
    range_image1: np.ndarray,
    range_image2: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute normalized difference between two range images

    Args:
        range_image1: First range image
        range_image2: Second range image
        threshold: Difference threshold in meters

    Returns:
        Fraction of pixels with difference > threshold
    """
    # Only compare valid pixels (both non-zero)
    valid_mask = (range_image1 > 0) & (range_image2 > 0)

    if valid_mask.sum() == 0:
        return 1.0  # Completely different

    diff = np.abs(range_image1 - range_image2)
    different_pixels = (diff[valid_mask] > threshold).sum()

    return different_pixels / valid_mask.sum()

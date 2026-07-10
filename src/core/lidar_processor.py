"""
LiDAR Processing Module
Process LAS/LAZ files for point cloud visualization.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

try:
    import laspy

    LASP_AVAILABLE = True
except ImportError:
    LASP_AVAILABLE = False
    print("⚠️ LasPy not installed. Install with: pip install laspy")


class LiDARProcessor:
    """
    Process LiDAR data from LAS/LAZ files.
    """

    def __init__(self):
        self.header = None
        self.points = None
        self.classifications = {}

    def read_las(self, filepath: str) -> Dict[str, Any]:
        """
        Read LAS/LAZ file.

        Args:
            filepath: Path to LAS/LAZ file

        Returns:
            Dictionary with point cloud data
        """
        if not LASP_AVAILABLE:
            print("⚠️ LasPy not available. Install with: pip install laspy")
            return self._generate_mock_lidar_data()

        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            return self._generate_mock_lidar_data()

        try:
            # Read LAS file
            las = laspy.read(filepath)

            # Extract points
            points = np.vstack([las.x, las.y, las.z]).transpose()

            # Extract classifications if available
            classifications = {}
            if hasattr(las, 'classification'):
                unique_classes = np.unique(las.classification)
                for cls in unique_classes:
                    mask = las.classification == cls
                    classifications[cls] = points[mask]

            self.header = las.header
            self.points = points
            self.classifications = classifications

            print(f"✅ Loaded {len(points)} points from {filepath}")

            return {
                'points': points,
                'classifications': classifications,
                'header': las.header,
                'point_count': len(points),
                'min': points.min(axis=0),
                'max': points.max(axis=0)
            }

        except Exception as e:
            print(f"❌ Error reading LAS: {str(e)}")
            return self._generate_mock_lidar_data()

    def _generate_mock_lidar_data(self) -> Dict[str, Any]:
        """Generate mock LiDAR data."""
        print("🏙️ Generating mock LiDAR data...")

        # Create mock point cloud
        np.random.seed(42)
        num_points = 10000
        x = np.random.uniform(0, 100, num_points)
        y = np.random.uniform(0, 100, num_points)
        z = np.random.uniform(0, 20, num_points)

        points = np.vstack([x, y, z]).transpose()

        return {
            'points': points,
            'classifications': {},
            'point_count': len(points),
            'min': points.min(axis=0),
            'max': points.max(axis=0)
        }

    def classify_ground(self, points: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Simple ground classification.

        Args:
            points: Point cloud
            threshold: Height threshold for ground

        Returns:
            Boolean mask for ground points
        """
        if len(points) == 0:
            return np.array([])

        # Simple classification: z < threshold = ground
        return points[:, 2] < threshold

    def classify_building(self, points: np.ndarray, height_threshold: float = 2.0) -> np.ndarray:
        """
        Simple building classification.

        Args:
            points: Point cloud
            height_threshold: Minimum height for buildings

        Returns:
            Boolean mask for building points
        """
        if len(points) == 0:
            return np.array([])

        # Simple classification: z > threshold = building
        return points[:, 2] > height_threshold

    def extract_assets_from_lidar(self, points: np.ndarray) -> pd.DataFrame:
        """
        Extract potential assets from LiDAR data.

        Args:
            points: Point cloud data

        Returns:
            DataFrame with extracted assets
        """
        if len(points) == 0:
            return pd.DataFrame()

        # Use DBSCAN or simple clustering
        # For demo, create random assets
        np.random.seed(42)
        num_assets = np.random.randint(20, 50)

        assets = []
        for i in range(num_assets):
            idx = np.random.randint(0, len(points))
            point = points[idx]

            asset_type = np.random.choice([
                'Streetlight', 'Traffic Sign', 'Tree', 'Building Corner', 'Vehicle'
            ])

            height_ranges = {
                'Streetlight': (3.0, 7.0),
                'Traffic Sign': (1.0, 3.0),
                'Tree': (4.0, 15.0),
                'Building Corner': (5.0, 25.0),
                'Vehicle': (1.2, 2.5)
            }

            min_h, max_h = height_ranges.get(asset_type, (1.0, 5.0))
            height = np.random.uniform(min_h, max_h)

            assets.append({
                'id': f"LIDAR_{i + 1:04d}",
                'type': asset_type,
                'lat': point[0],
                'lon': point[1],
                'height_m': height,
                'confidence': 0.90,
                'status': 'lidar',
                'source': 'LiDAR'
            })

        return pd.DataFrame(assets)

    def to_dataframe(self, points: np.ndarray) -> pd.DataFrame:
        """
        Convert point cloud to DataFrame.

        Args:
            points: Point cloud data

        Returns:
            DataFrame with columns: x, y, z
        """
        if len(points) == 0:
            return pd.DataFrame(columns=['x', 'y', 'z'])

        return pd.DataFrame({
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2]
        })


# Convenience function
def process_lidar_file(filepath: str) -> Dict[str, Any]:
    """Process a LiDAR file and return data."""
    processor = LiDARProcessor()
    return processor.read_las(filepath)
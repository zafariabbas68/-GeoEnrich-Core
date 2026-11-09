

import numpy as np
import pandas as pd
import random
import os
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class AssetDefinition:
    """Definition of an infrastructure asset type."""
    name: str
    height_range: Tuple[float, float]
    offset_range: Tuple[float, float]
    point_count_range: Tuple[int, int]
    color: str
    description: str


class MobileMappingGenerator:
    """
    Generates synthetic mobile mapping datasets.

    Creates:
    - Road network (centerline with curves)
    - Infrastructure assets (streetlights, signs, trees, etc.)
    - Simulated LiDAR point cloud for each asset
    - Geospatial relationships between assets
    """

    # Predefined asset types with realistic parameters
    ASSET_TYPES = {
        'Streetlight': AssetDefinition(
            name='Streetlight',
            height_range=(3.0, 8.0),
            offset_range=(0.5, 1.5),
            point_count_range=(50, 200),
            color='#FF6B6B',
            description='Vertical pole with light fixture'
        ),
        'Traffic Sign': AssetDefinition(
            name='Traffic Sign',
            height_range=(1.0, 3.5),
            offset_range=(0.3, 1.0),
            point_count_range=(20, 80),
            color='#4ECDC4',
            description='Road sign on vertical post'
        ),
        'Tree': AssetDefinition(
            name='Tree',
            height_range=(4.0, 20.0),
            offset_range=(1.0, 4.0),
            point_count_range=(100, 500),
            color='#45B7D1',
            description='Vertical trunk with canopy'
        ),
        'Manhole': AssetDefinition(
            name='Manhole',
            height_range=(0.0, 0.3),
            offset_range=(0.0, 0.5),
            point_count_range=(10, 40),
            color='#96CEB4',
            description='Ground-level circular cover'
        ),
        'Fire Hydrant': AssetDefinition(
            name='Fire Hydrant',
            height_range=(0.5, 1.2),
            offset_range=(0.2, 0.8),
            point_count_range=(30, 100),
            color='#FFEAA7',
            description='Vertical red post'
        ),
        'Building Corner': AssetDefinition(
            name='Building Corner',
            height_range=(3.0, 50.0),
            offset_range=(2.0, 6.0),
            point_count_range=(200, 800),
            color='#DDA0DD',
            description='Vertical wall corner'
        ),
        'Vehicle': AssetDefinition(
            name='Vehicle',
            height_range=(1.0, 3.0),
            offset_range=(1.0, 2.5),
            point_count_range=(100, 300),
            color='#FF9FF3',
            description='Parked car or truck'
        )
    }

    def __init__(self, center_lat: float = 52.3676, center_lon: float = 4.9041):
        """
        Initialize generator with center coordinates.

        Args:
            center_lat: Center latitude (default: Amsterdam Dam Square)
            center_lon: Center longitude (default: Amsterdam Dam Square)
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.assets = []
        self.road_points = []
        self.point_cloud = []

    def generate_road_network(self,
                              length_km: float = 0.5,
                              segments: int = 6) -> List[Tuple[float, float]]:
        """
        Generate a realistic road network with curves.

        Args:
            length_km: Road length in kilometers
            segments: Number of segments for curvature

        Returns:
            List of (lat, lon) points forming the road
        """
        # Convert km to degrees (approximate)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(math.radians(self.center_lat))
        length_deg = length_km / km_per_deg_lat

        road_points = []
        current_lat = self.center_lat
        current_lon = self.center_lon

        for i in range(segments + 1):
            # Progress along road
            t = i / segments

            # Base path (straight line with slight curve)
            lat_offset = t * length_deg

            # Add sinusoidal curve for realism
            curve_amplitude = 0.0002 * length_km  # 0.2m per km
            lon_offset = (t * length_deg * 0.3 +
                          curve_amplitude * math.sin(t * 2.5 * math.pi))

            point = (
                current_lat + lat_offset,
                current_lon + lon_offset
            )
            road_points.append(point)

        self.road_points = road_points
        return road_points

    def generate_assets(self,
                        num_assets: int = 50,
                        asset_types: List[str] = None) -> pd.DataFrame:
        """
        Generate infrastructure assets along the road.

        Args:
            num_assets: Number of assets to generate
            asset_types: List of asset types to use (default: all)

        Returns:
            DataFrame with asset information
        """
        if asset_types is None:
            asset_types = list(self.ASSET_TYPES.keys())

        assets = []

        for i in range(num_assets):
            # Select random asset type
            asset_name = random.choice(asset_types)
            asset_def = self.ASSET_TYPES[asset_name]

            # Position along road
            if len(self.road_points) < 2:
                self.generate_road_network()

            road_idx = random.randint(0, len(self.road_points) - 2)
            road_pos = random.random()

            # Interpolate position along road segment
            lat1, lon1 = self.road_points[road_idx]
            lat2, lon2 = self.road_points[road_idx + 1]

            base_lat = lat1 + (lat2 - lat1) * road_pos
            base_lon = lon1 + (lon2 - lon1) * road_pos

            # Add offset perpendicular to road
            offset_dist = random.uniform(*asset_def.offset_range)

            # Direction perpendicular to road
            dx = lat2 - lat1
            dy = lon2 - lon1
            length = math.sqrt(dx ** 2 + dy ** 2)
            if length > 0:
                # Perpendicular direction
                perp_lat = -dy / length
                perp_lon = dx / length

                # Random side of road (left or right)
                side = random.choice([-1, 1])

                # Convert offset to degrees
                offset_lat = side * offset_dist * perp_lat / 111.32
                offset_lon = side * offset_dist * perp_lon / (111.32 * math.cos(math.radians(lat1)))
            else:
                offset_lat = 0
                offset_lon = 0

            lat = base_lat + offset_lat
            lon = base_lon + offset_lon

            # Generate height
            height = random.uniform(*asset_def.height_range)

            # Generate asset ID
            asset_id = f"ASSET_{i + 1:04d}_{asset_name[:3]}"

            # Generate point cloud for this asset
            point_cloud = self._generate_asset_point_cloud(
                lat, lon, height, asset_def
            )

            assets.append({
                'id': asset_id,
                'type': asset_name,
                'lat': lat,
                'lon': lon,
                'height_m': height,
                'point_cloud': point_cloud,
                'confidence': random.uniform(0.75, 0.98),
                'status': 'detected'
            })

        self.assets = assets
        return pd.DataFrame(assets)

    def _generate_asset_point_cloud(self,
                                    lat: float,
                                    lon: float,
                                    height: float,
                                    asset_def: AssetDefinition) -> List[Tuple[float, float, float]]:
        """
        Generate simulated LiDAR points for an asset.

        Args:
            lat, lon: Asset location
            height: Asset height in meters
            asset_def: Asset definition

        Returns:
            List of (lat, lon, z) points
        """
        points = []
        num_points = random.randint(*asset_def.point_count_range)

        # Base resolution (meters per degree)
        m_per_deg_lat = 111320
        m_per_deg_lon = 111320 * math.cos(math.radians(lat))

        for _ in range(num_points):
            # Position spread (meters)
            spread = random.uniform(0.05, 0.3)

            # Convert spread to degrees
            lat_offset = random.uniform(-spread, spread) / m_per_deg_lat
            lon_offset = random.uniform(-spread, spread) / m_per_deg_lon

            # Z value based on asset type
            if asset_def.name in ['Streetlight', 'Tree', 'Building Corner']:
                # Points distributed along height
                z = random.uniform(0, height)
            elif asset_def.name == 'Traffic Sign':
                # Points concentrated near top
                z = random.uniform(height * 0.6, height)
            elif asset_def.name == 'Manhole':
                # Flat, near ground
                z = random.uniform(0, 0.2)
            else:
                # Mixed distribution
                z = random.uniform(0, height)

            points.append((
                lat + lat_offset,
                lon + lon_offset,
                z
            ))

        return points

    def generate_full_dataset(self, output_dir: str = "outputs/sample_data"):
        """
        Generate complete dataset and save to files.

        Args:
            output_dir: Directory to save generated data

        Returns:
            Dictionary with file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        # Generate road
        road_points = self.generate_road_network(length_km=0.5)
        road_df = pd.DataFrame(road_points, columns=['lat', 'lon'])
        road_df['point_id'] = road_df.index
        road_path = os.path.join(output_dir, 'road_network.csv')
        road_df.to_csv(road_path, index=False)

        # Generate assets
        assets_df = self.generate_assets(num_assets=50)
        assets_path = os.path.join(output_dir, 'assets.csv')
        assets_df.to_csv(assets_path, index=False)

        # Generate point cloud
        all_points = []
        for _, asset in assets_df.iterrows():
            for point in asset['point_cloud']:
                all_points.append([
                    point[0], point[1], point[2],
                    asset['id'], asset['type']
                ])

        point_cloud_array = np.array(all_points, dtype=object)
        point_cloud_path = os.path.join(output_dir, 'point_cloud.npy')
        np.save(point_cloud_path, point_cloud_array)

        # Also save as CSV
        pc_df = pd.DataFrame(all_points, columns=['lat', 'lon', 'z', 'asset_id', 'asset_type'])
        pc_csv_path = os.path.join(output_dir, 'point_cloud.csv')
        pc_df.to_csv(pc_csv_path, index=False)

        return {
            'road_path': road_path,
            'assets_path': assets_path,
            'point_cloud_path': point_cloud_path,
            'point_cloud_csv': pc_csv_path
        }


# Convenience function
def generate_sample_data(output_dir: str = "outputs/sample_data"):
    """Generate a sample dataset for testing."""
    generator = MobileMappingGenerator()
    return generator.generate_full_dataset(output_dir)


if __name__ == "__main__":
    # Test generation
    print("🚀 Generating sample mobile mapping data...")
    generator = MobileMappingGenerator()
    files = generator.generate_full_dataset()
    print("✅ Generated files:")
    for name, path in files.items():
        print(f"   - {name}: {path}")

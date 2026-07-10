"""
Spatial Analytics Module
Performs geospatial analysis on mobile mapping data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class ProximityResult:
    """Container for proximity analysis results."""
    asset_id: str
    asset_type: str
    lat: float
    lon: float
    height_m: float
    neighbor_count: int
    neighbors: List[Dict[str, Any]]
    is_isolated: bool
    mean_distance_km: float


class SpatialAnalytics:
    """
    Comprehensive spatial analysis engine for mobile mapping data.
    """

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in km

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @staticmethod
    def calculate_proximity(assets_df: pd.DataFrame,
                            max_distance_km: float = 0.1) -> pd.DataFrame:
        """
        Calculate proximity between assets.

        Args:
            assets_df: DataFrame with asset locations
            max_distance_km: Maximum distance for proximity

        Returns:
            DataFrame with proximity information
        """
        results = []

        for i, asset1 in assets_df.iterrows():
            neighbors = []

            for j, asset2 in assets_df.iterrows():
                if i == j:
                    continue

                distance = SpatialAnalytics.haversine(
                    asset1['lat'], asset1['lon'],
                    asset2['lat'], asset2['lon']
                )

                if distance <= max_distance_km:
                    neighbors.append({
                        'neighbor_id': asset2['id'],
                        'neighbor_type': asset2['type'],
                        'distance_km': round(distance, 4)
                    })

            mean_dist = 0
            if neighbors:
                mean_dist = sum(n['distance_km'] for n in neighbors) / len(neighbors)

            results.append({
                'asset_id': asset1['id'],
                'asset_type': asset1['type'],
                'lat': asset1['lat'],
                'lon': asset1['lon'],
                'height_m': asset1.get('height_m', 0),
                'neighbor_count': len(neighbors),
                'neighbors': neighbors,
                'is_isolated': len(neighbors) == 0,
                'mean_distance_km': round(mean_dist, 4)
            })

        return pd.DataFrame(results)

    @staticmethod
    def cluster_assets(assets_df: pd.DataFrame,
                       max_cluster_distance_km: float = 0.15) -> pd.DataFrame:
        """
        Cluster assets based on spatial proximity.

        Args:
            assets_df: DataFrame with asset locations
            max_cluster_distance_km: Maximum distance for clustering

        Returns:
            DataFrame with cluster assignments
        """
        # Simple DBSCAN-like clustering
        points = assets_df[['lat', 'lon']].values.tolist()
        n_points = len(points)

        # Initialize clusters
        clusters = [-1] * n_points
        current_cluster = 0

        for i in range(n_points):
            if clusters[i] != -1:
                continue

            # Start new cluster
            cluster_points = [i]
            clusters[i] = current_cluster

            # Expand cluster
            for j in range(i + 1, n_points):
                if clusters[j] != -1:
                    continue

                distance = SpatialAnalytics.haversine(
                    points[i][0], points[i][1],
                    points[j][0], points[j][1]
                )

                if distance <= max_cluster_distance_km:
                    clusters[j] = current_cluster
                    cluster_points.append(j)

            current_cluster += 1

        # Map clusters back to assets
        clustered_data = []
        for idx, cluster_id in enumerate(clusters):
            clustered_data.append({
                'asset_id': assets_df.iloc[idx]['id'],
                'asset_type': assets_df.iloc[idx]['type'],
                'lat': assets_df.iloc[idx]['lat'],
                'lon': assets_df.iloc[idx]['lon'],
                'height_m': assets_df.iloc[idx].get('height_m', 0),
                'cluster_id': cluster_id,
                'cluster_size': clusters.count(cluster_id)
            })

        return pd.DataFrame(clustered_data)

    @staticmethod
    def calculate_buffer_zones(assets_df: pd.DataFrame,
                               buffer_km: float = 0.05) -> pd.DataFrame:
        """
        Calculate buffer zones around assets.

        Args:
            assets_df: DataFrame with asset locations
            buffer_km: Buffer radius in kilometers

        Returns:
            DataFrame with buffer zone information
        """
        buffer_data = []

        for _, asset in assets_df.iterrows():
            # Buffer area (circle area on sphere)
            buffer_area_km2 = math.pi * (buffer_km ** 2)

            buffer_data.append({
                'asset_id': asset['id'],
                'asset_type': asset['type'],
                'lat': asset['lat'],
                'lon': asset['lon'],
                'buffer_radius_km': buffer_km,
                'buffer_area_km2': round(buffer_area_km2, 4),
                'height_m': asset.get('height_m', 0)
            })

        return pd.DataFrame(buffer_data)

    @staticmethod
    def analyze_distribution(assets_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze spatial distribution of assets.

        Args:
            assets_df: DataFrame with asset locations

        Returns:
            Dictionary with distribution statistics
        """
        points = assets_df[['lat', 'lon']].values.tolist()
        n_points = len(points)

        if n_points == 0:
            return {}

        # Calculate centroid
        centroid_lat = sum(p[0] for p in points) / n_points
        centroid_lon = sum(p[1] for p in points) / n_points

        # Calculate pairwise distances
        distances = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = SpatialAnalytics.haversine(
                    points[i][0], points[i][1],
                    points[j][0], points[j][1]
                )
                distances.append(dist)

        # Asset type distribution
        type_counts = assets_df['type'].value_counts().to_dict()

        return {
            'total_assets': n_points,
            'asset_types': type_counts,
            'unique_types': len(type_counts),
            'centroid_lat': round(centroid_lat, 6),
            'centroid_lon': round(centroid_lon, 6),
            'mean_distance_km': round(np.mean(distances), 4) if distances else 0,
            'max_distance_km': round(max(distances), 4) if distances else 0,
            'std_distance_km': round(np.std(distances), 4) if distances else 0
        }
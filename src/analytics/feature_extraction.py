"""
Feature Extraction Module
Extracts geometric and statistical features from point cloud data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.spatial import KDTree
from scipy.stats import skew, kurtosis


class FeatureExtractor:
    """
    Extracts features from point cloud data for asset classification.
    """

    @staticmethod
    def extract_point_cloud_features(points: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive features from point cloud.

        Args:
            points: Numpy array with point cloud data (lat, lon, z)

        Returns:
            Dictionary of extracted features
        """
        if len(points) == 0:
            return {
                'num_points': 0,
                'height_min': 0,
                'height_max': 0,
                'height_mean': 0,
                'height_std': 0,
                'height_range': 0,
                'spread': 0,
                'density': 0,
                'skewness': 0,
                'kurtosis': 0,
                'point_density': 0
            }

        # Extract Z values
        z_values = points[:, 2].astype(float)

        # Height statistics
        height_min = np.min(z_values)
        height_max = np.max(z_values)
        height_mean = np.mean(z_values)
        height_std = np.std(z_values)
        height_range = height_max - height_min

        # Spatial spread (horizontal)
        lat_std = np.std(points[:, 0].astype(float))
        lon_std = np.std(points[:, 1].astype(float))
        spread = np.sqrt(lat_std ** 2 + lon_std ** 2) * 111320  # Convert to meters

        # Density (points per cubic meter)
        if height_range > 0:
            volume = spread * spread * height_range
            density = len(points) / volume if volume > 0 else 0
        else:
            density = 0

        # Distribution statistics
        skewness = skew(z_values) if len(z_values) > 2 else 0
        kurt = kurtosis(z_values) if len(z_values) > 3 else 0

        return {
            'num_points': len(points),
            'height_min': round(height_min, 2),
            'height_max': round(height_max, 2),
            'height_mean': round(height_mean, 2),
            'height_std': round(height_std, 2),
            'height_range': round(height_range, 2),
            'spread': round(spread, 2),
            'density': round(density, 4),
            'skewness': round(skewness, 3),
            'kurtosis': round(kurt, 3),
            'point_density': round(len(points) / (spread * spread) if spread > 0 else 0, 2)
        }

    @staticmethod
    def extract_spatial_features(assets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spatial relationship features between assets.

        Args:
            assets_df: DataFrame with asset locations

        Returns:
            DataFrame with spatial features
        """
        from src.analytics.spatial_analytics import SpatialAnalytics

        features = []

        for i, asset1 in assets_df.iterrows():
            # Find nearest neighbor
            min_distance = float('inf')
            nearest_id = None
            nearest_type = None

            for j, asset2 in assets_df.iterrows():
                if i == j:
                    continue

                distance = SpatialAnalytics.haversine(
                    asset1['lat'], asset1['lon'],
                    asset2['lat'], asset2['lon']
                )

                if distance < min_distance:
                    min_distance = distance
                    nearest_id = asset2['id']
                    nearest_type = asset2['type']

            features.append({
                'asset_id': asset1['id'],
                'nearest_distance_km': round(min_distance, 4) if min_distance < float('inf') else None,
                'nearest_id': nearest_id,
                'nearest_type': nearest_type
            })

        return pd.DataFrame(features)

    @staticmethod
    def extract_shape_features(points: np.ndarray) -> Dict[str, Any]:
        """
        Extract shape classification features.

        Args:
            points: Numpy array with point cloud data

        Returns:
            Shape features dictionary
        """
        if len(points) < 10:
            return {'shape': 'unknown', 'confidence': 0.0}

        features = FeatureExtractor.extract_point_cloud_features(points)

        height_range = features['height_range']
        spread = features['spread']
        density = features['density']

        # Shape classification based on geometry
        shape_score = {}

        # Vertical (tall, narrow)
        if height_range > 2 and spread < 1:
            vertical_score = min(1.0, (height_range / 5) * 0.8 + 0.2)
            shape_score['vertical'] = vertical_score
        else:
            shape_score['vertical'] = 0.1

        # Flat (low height, wide spread)
        if height_range < 0.5 and spread > 0.3:
            flat_score = min(1.0, (spread / 1) * 0.7 + 0.3)
            shape_score['flat'] = flat_score
        else:
            shape_score['flat'] = 0.1

        # Spherical (balanced height and spread)
        if 0.5 < height_range < 3 and 0.5 < spread < 2:
            spherical_score = 0.7
            shape_score['spherical'] = spherical_score
        else:
            shape_score['spherical'] = 0.1

        # Blocky (large spread, significant height)
        if height_range > 2 and spread > 1.5:
            blocky_score = min(1.0, (height_range / 10) * 0.5 + 0.3)
            shape_score['blocky'] = blocky_score
        else:
            shape_score['blocky'] = 0.1

        # Find best shape
        best_shape = max(shape_score, key=shape_score.get)
        best_confidence = shape_score[best_shape]

        return {
            'shape': best_shape,
            'confidence': round(best_confidence, 3),
            'shape_scores': shape_score,
            'height_range': height_range,
            'spread': spread,
            'density': density
        }
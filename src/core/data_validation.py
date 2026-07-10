"""
Data Validation Module
QA/QC checks for mobile mapping data.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any


class DataValidator:
    """
    Validates mobile mapping data for quality and completeness.
    """

    @staticmethod
    def validate_assets(assets_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate asset data for completeness and quality.

        Args:
            assets_df: DataFrame with asset information

        Returns:
            Validation results with errors and warnings
        """
        errors = []
        warnings = []

        # Check for required columns
        required_cols = ['id', 'type', 'lat', 'lon', 'height_m']
        missing_cols = [col for col in required_cols if col not in assets_df.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
            return {'errors': errors, 'warnings': warnings, 'is_valid': False}

        # Check for duplicate IDs
        duplicate_ids = assets_df['id'][assets_df['id'].duplicated()].tolist()
        if duplicate_ids:
            errors.append(f"Duplicate IDs found: {duplicate_ids[:5]}...")

        # Check for missing values
        for col in required_cols:
            missing = assets_df[col].isnull().sum()
            if missing > 0:
                warnings.append(f"{col}: {missing} missing values")

        # Check coordinate ranges
        invalid_lat = assets_df[(assets_df['lat'] < -90) | (assets_df['lat'] > 90)]
        invalid_lon = assets_df[(assets_df['lon'] < -180) | (assets_df['lon'] > 180)]

        if len(invalid_lat) > 0:
            errors.append(f"Invalid latitude values: {len(invalid_lat)}")
        if len(invalid_lon) > 0:
            errors.append(f"Invalid longitude values: {len(invalid_lon)}")

        # Check height values
        negative_height = assets_df[assets_df['height_m'] < 0]
        if len(negative_height) > 0:
            errors.append(f"Negative height values: {len(negative_height)}")

        # Check for valid asset types
        valid_types = [
            'Streetlight', 'Traffic Sign', 'Tree', 'Manhole',
            'Fire Hydrant', 'Building Corner', 'Vehicle'
        ]
        invalid_types = assets_df[~assets_df['type'].isin(valid_types)]
        if len(invalid_types) > 0:
            warnings.append(f"Unrecognized asset types: {invalid_types['type'].unique().tolist()}")

        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0,
            'total_assets': len(assets_df),
            'asset_types': assets_df['type'].value_counts().to_dict()
        }

    @staticmethod
    def validate_point_cloud(point_cloud: np.ndarray) -> Dict[str, Any]:
        """
        Validate point cloud data.

        Args:
            point_cloud: Numpy array with point cloud data

        Returns:
            Validation results
        """
        errors = []
        warnings = []

        if len(point_cloud) == 0:
            errors.append("Point cloud is empty")
            return {'errors': errors, 'warnings': warnings, 'is_valid': False}

        # Check for required columns
        if point_cloud.shape[1] < 3:
            errors.append(f"Point cloud has {point_cloud.shape[1]} columns, expected at least 3 (lat, lon, z)")

        # Check for NaN values
        nan_count = np.isnan(point_cloud[:, :3].astype(float)).sum()
        if nan_count > 0:
            warnings.append(f"{nan_count} NaN values found in coordinates")

        # Check coordinate ranges
        lat_col = point_cloud[:, 0].astype(float)
        lon_col = point_cloud[:, 1].astype(float)

        invalid_lat = np.sum((lat_col < -90) | (lat_col > 90))
        invalid_lon = np.sum((lon_col < -180) | (lon_col > 180))

        if invalid_lat > 0:
            errors.append(f"{invalid_lat} invalid latitude values")
        if invalid_lon > 0:
            errors.append(f"{invalid_lon} invalid longitude values")

        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0,
            'total_points': len(point_cloud),
            'dimensions': point_cloud.shape[1]
        }

    @staticmethod
    def validate_road_network(road_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate road network data.

        Args:
            road_df: DataFrame with road points

        Returns:
            Validation results
        """
        errors = []
        warnings = []

        if len(road_df) == 0:
            errors.append("Road network is empty")
            return {'errors': errors, 'warnings': warnings, 'is_valid': False}

        # Check for required columns
        if 'lat' not in road_df.columns or 'lon' not in road_df.columns:
            errors.append("Road network missing 'lat' or 'lon' columns")

        # Check for minimum points
        if len(road_df) < 2:
            warnings.append(f"Road network has only {len(road_df)} points")

        return {
            'errors': errors,
            'warnings': warnings,
            'is_valid': len(errors) == 0,
            'total_points': len(road_df)
        }

    @staticmethod
    def generate_validation_report(assets_df: pd.DataFrame,
                                   point_cloud: np.ndarray,
                                   road_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate complete validation report.

        Args:
            assets_df: Asset DataFrame
            point_cloud: Point cloud array
            road_df: Road network DataFrame

        Returns:
            Complete validation report
        """
        report = {
            'assets': DataValidator.validate_assets(assets_df),
            'point_cloud': DataValidator.validate_point_cloud(point_cloud),
            'road_network': DataValidator.validate_road_network(road_df),
            'overall_valid': True
        }

        # Check overall validity
        report['overall_valid'] = (
                report['assets']['is_valid'] and
                report['point_cloud']['is_valid'] and
                report['road_network']['is_valid']
        )

        return report


# Quick test
if __name__ == "__main__":
    # Test with sample data
    try:
        from src.core.data_ingestion import DataIngestion
        from src.core.data_generation import generate_sample_data

        # Generate sample data
        generate_sample_data()

        # Load and validate
        data = DataIngestion.load_dataset()

        report = DataValidator.generate_validation_report(
            data['assets'],
            data['point_cloud'],
            data['road_network']
        )

        print("📊 Validation Report:")
        print(f"  Overall Valid: {report['overall_valid']}")
        print(f"  Assets: {report['assets']['is_valid']} ({len(data['assets'])} assets)")
        print(f"  Point Cloud: {report['point_cloud']['is_valid']} ({len(data['point_cloud'])} points)")
        print(f"  Road Network: {report['road_network']['is_valid']}")

        if report['assets']['errors']:
            print(f"  Errors: {report['assets']['errors']}")
        if report['assets']['warnings']:
            print(f"  Warnings: {report['assets']['warnings']}")

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
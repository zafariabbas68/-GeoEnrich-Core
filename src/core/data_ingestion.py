"""
Data Ingestion Module
Loads point cloud and asset data from various formats.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any


class DataIngestion:
    """
    Handles loading of mobile mapping data from various sources.
    """

    @staticmethod
    def load_assets_from_csv(filepath: str) -> pd.DataFrame:
        """
        Load asset data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with asset information
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        # Validate required columns
        required_columns = ['id', 'type', 'lat', 'lon']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")

        return df

    @staticmethod
    def load_point_cloud_npy(filepath: str) -> np.ndarray:
        """
        Load point cloud from numpy .npy file.

        Args:
            filepath: Path to .npy file

        Returns:
            Numpy array with point cloud data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        return np.load(filepath, allow_pickle=True)

    @staticmethod
    def load_point_cloud_csv(filepath: str) -> np.ndarray:
        """
        Load point cloud from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Numpy array with point cloud data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        # Validate required columns
        required_columns = ['lat', 'lon', 'z']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")

        # Convert to numpy array
        points = df[['lat', 'lon', 'z']].values

        # Add additional columns if they exist
        if 'asset_id' in df.columns and 'asset_type' in df.columns:
            return df[['lat', 'lon', 'z', 'asset_id', 'asset_type']].values

        return points

    @staticmethod
    def load_road_network(filepath: str) -> pd.DataFrame:
        """
        Load road network data from CSV.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with road points
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        # Validate required columns
        required_columns = ['lat', 'lon']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")

        return df

    @staticmethod
    def load_dataset(data_dir: str = "outputs/sample_data") -> Dict[str, Any]:
        """
        Load complete dataset from directory.

        Args:
            data_dir: Directory containing data files

        Returns:
            Dictionary with all loaded data
        """
        assets_path = os.path.join(data_dir, 'assets.csv')
        road_path = os.path.join(data_dir, 'road_network.csv')
        pc_npy_path = os.path.join(data_dir, 'point_cloud.npy')
        pc_csv_path = os.path.join(data_dir, 'point_cloud.csv')

        result = {}

        # Load assets
        if os.path.exists(assets_path):
            result['assets'] = DataIngestion.load_assets_from_csv(assets_path)
        else:
            result['assets'] = None

        # Load road network
        if os.path.exists(road_path):
            result['road_network'] = DataIngestion.load_road_network(road_path)
        else:
            result['road_network'] = None

        # Load point cloud (prefer .npy over .csv)
        if os.path.exists(pc_npy_path):
            result['point_cloud'] = DataIngestion.load_point_cloud_npy(pc_npy_path)
        elif os.path.exists(pc_csv_path):
            result['point_cloud'] = DataIngestion.load_point_cloud_csv(pc_csv_path)
        else:
            result['point_cloud'] = None

        return result


# Quick test
if __name__ == "__main__":
    # Test loading
    try:
        data = DataIngestion.load_dataset()
        if data['assets'] is not None:
            print(f"✅ Loaded {len(data['assets'])} assets")
        if data['point_cloud'] is not None:
            print(f"✅ Loaded {len(data['point_cloud'])} point cloud points")
        if data['road_network'] is not None:
            print(f"✅ Loaded {len(data['road_network'])} road points")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Run `python scripts/run_full_pipeline.py` first to generate data.")
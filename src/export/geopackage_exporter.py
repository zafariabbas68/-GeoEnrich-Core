"""
GeoPackage Export Module
Exports data to GeoPackage format for GIS applications.
"""

import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def export_to_geopackage(assets_df: pd.DataFrame,
                         output_path: str = "outputs/assets.gpkg") -> str:
    """
    Export assets to GeoPackage format.

    Args:
        assets_df: DataFrame with asset data
        output_path: Output file path

    Returns:
        Path to created file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create geometry column
    geometry = [Point(xy) for xy in zip(assets_df['lon'], assets_df['lat'])]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        assets_df,
        geometry=geometry,
        crs="EPSG:4326"
    )

    # Export
    gdf.to_file(output_path, driver="GPKG")

    return output_path


def export_recognition_to_geopackage(recognition_df: pd.DataFrame,
                                     assets_df: pd.DataFrame,
                                     output_path: str = "outputs/recognition.gpkg") -> str:
    """
    Export recognition results to GeoPackage.

    Args:
        recognition_df: Recognition results
        assets_df: Original asset data
        output_path: Output file path

    Returns:
        Path to created file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Merge recognition with asset locations
    merged = assets_df.merge(recognition_df, left_on='id', right_on='id')

    # Create geometry
    geometry = [Point(xy) for xy in zip(merged['lon'], merged['lat'])]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        merged,
        geometry=geometry,
        crs="EPSG:4326"
    )

    # Export
    gdf.to_file(output_path, driver="GPKG")

    return output_path
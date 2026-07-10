"""
Map Layer Generation Module
Creates interactive map layers for the dashboard.
"""

import folium
import pandas as pd
from folium.plugins import MarkerCluster


def create_asset_cluster_map(assets_df: pd.DataFrame,
                             road_df: pd.DataFrame = None) -> folium.Map:
    """
    Create map with asset clustering.

    Args:
        assets_df: DataFrame with asset data
        road_df: DataFrame with road network

    Returns:
        Folium Map
    """
    center_lat = assets_df['lat'].mean()
    center_lon = assets_df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Add road network
    if road_df is not None:
        road_points = [[row['lat'], row['lon']] for _, row in road_df.iterrows()]
        folium.PolyLine(road_points, color='gray', weight=2, opacity=0.5).add_to(m)

    # Create marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Add assets
    for _, asset in assets_df.iterrows():
        popup = f"""
        <b>{asset['type']}</b><br>
        ID: {asset['id']}<br>
        Height: {asset.get('height_m', 0):.2f}m
        """

        folium.CircleMarker(
            location=[asset['lat'], asset['lon']],
            radius=6,
            popup=popup,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7
        ).add_to(marker_cluster)

    return m


def create_asset_heatmap(assets_df: pd.DataFrame) -> folium.Map:
    """
    Create heatmap of asset density.

    Args:
        assets_df: DataFrame with asset data

    Returns:
        Folium Map
    """
    from folium.plugins import HeatMap

    center_lat = assets_df['lat'].mean()
    center_lon = assets_df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Prepare data for heatmap
    heat_data = [[row['lat'], row['lon']] for _, row in assets_df.iterrows()]

    HeatMap(heat_data).add_to(m)

    return m


def create_layer_control_map(assets_df: pd.DataFrame,
                             road_df: pd.DataFrame = None) -> folium.Map:
    """
    Create map with layer control.

    Args:
        assets_df: DataFrame with asset data
        road_df: DataFrame with road network

    Returns:
        Folium Map
    """
    center_lat = assets_df['lat'].mean()
    center_lon = assets_df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Base layer
    folium.TileLayer('CartoDB positron').add_to(m)

    # Asset layer
    asset_group = folium.FeatureGroup(name='Assets')
    for _, asset in assets_df.iterrows():
        popup = f"""
        <b>{asset['type']}</b><br>
        ID: {asset['id']}<br>
        Height: {asset.get('height_m', 0):.2f}m
        """
        folium.CircleMarker(
            location=[asset['lat'], asset['lon']],
            radius=6,
            popup=popup,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7
        ).add_to(asset_group)
    asset_group.add_to(m)

    # Road network
    if road_df is not None:
        road_group = folium.FeatureGroup(name='Road Network')
        road_points = [[row['lat'], row['lon']] for _, row in road_df.iterrows()]
        folium.PolyLine(road_points, color='gray', weight=2, opacity=0.5).add_to(road_group)
        road_group.add_to(m)

    folium.LayerControl().add_to(m)

    return m


def create_asset_type_map(assets_df: pd.DataFrame,
                          road_df: pd.DataFrame = None) -> folium.Map:
    """
    Create map with color-coded asset types.

    Args:
        assets_df: DataFrame with asset data
        road_df: DataFrame with road network

    Returns:
        Folium Map
    """
    center_lat = assets_df['lat'].mean()
    center_lon = assets_df['lon'].mean()

    # Asset colors
    asset_colors = {
        'Streetlight': '#FF6B6B',
        'Traffic Sign': '#4ECDC4',
        'Tree': '#45B7D1',
        'Manhole': '#96CEB4',
        'Fire Hydrant': '#FFEAA7',
        'Building Corner': '#DDA0DD',
        'Vehicle': '#FF9FF3'
    }

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Add road network
    if road_df is not None:
        road_points = [[row['lat'], row['lon']] for _, row in road_df.iterrows()]
        folium.PolyLine(road_points, color='gray', weight=2, opacity=0.5).add_to(m)

    # Add assets
    for _, asset in assets_df.iterrows():
        color = asset_colors.get(asset['type'], '#95A5A6')
        popup = f"""
        <b>{asset['type']}</b><br>
        ID: {asset['id']}<br>
        Height: {asset.get('height_m', 0):.2f}m
        """

        folium.CircleMarker(
            location=[asset['lat'], asset['lon']],
            radius=6,
            popup=popup,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(m)

    return m
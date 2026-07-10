"""
Chart Generation Module
Creates visualizations for the dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_asset_type_pie(assets_df: pd.DataFrame,
                          colors: dict = None) -> go.Figure:
    """
    Create pie chart of asset type distribution.

    Args:
        assets_df: DataFrame with asset data
        colors: Dictionary of color mapping

    Returns:
        Plotly Figure
    """
    counts = assets_df['type'].value_counts()

    if colors is None:
        colors = {}

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Asset Type Distribution",
        color=counts.index,
        color_discrete_map=colors,
        hole=0.3
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)

    return fig


def create_height_bar_chart(assets_df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart of asset heights.

    Args:
        assets_df: DataFrame with asset data

    Returns:
        Plotly Figure
    """
    fig = px.bar(
        assets_df,
        x='id',
        y='height_m',
        color='type',
        title="Asset Heights"
    )

    fig.update_layout(
        height=300,
        xaxis_title="Asset ID",
        yaxis_title="Height (m)",
        showlegend=True
    )

    return fig


def create_neighbor_distribution(proximity_df: pd.DataFrame) -> go.Figure:
    """
    Create histogram of neighbor counts.

    Args:
        proximity_df: Proximity analysis results

    Returns:
        Plotly Figure
    """
    neighbor_counts = proximity_df['neighbor_count'].value_counts().sort_index()

    fig = px.bar(
        x=neighbor_counts.index,
        y=neighbor_counts.values,
        labels={'x': 'Number of Neighbors', 'y': 'Asset Count'},
        title="Neighbor Distribution"
    )

    fig.update_layout(height=250)

    return fig


def create_confidence_histogram(recognition_df: pd.DataFrame) -> go.Figure:
    """
    Create histogram of recognition confidence.

    Args:
        recognition_df: Recognition results

    Returns:
        Plotly Figure
    """
    fig = px.histogram(
        recognition_df,
        x='confidence',
        nbins=20,
        title="Recognition Confidence Distribution"
    )

    fig.update_layout(height=250)

    return fig


def create_confidence_by_type(recognition_df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart of confidence by asset type.

    Args:
        recognition_df: Recognition results

    Returns:
        Plotly Figure
    """
    avg_conf = recognition_df.groupby('recognized_type')['confidence'].mean().sort_values(ascending=False)

    fig = px.bar(
        x=avg_conf.values,
        y=avg_conf.index,
        orientation='h',
        title="Average Confidence by Type",
        color=avg_conf.values,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(height=200)

    return fig


def create_scatter_map(assets_df: pd.DataFrame) -> go.Figure:
    """
    Create scatter plot on map.

    Args:
        assets_df: DataFrame with asset data

    Returns:
        Plotly Figure
    """
    fig = px.scatter_mapbox(
        assets_df,
        lat='lat',
        lon='lon',
        color='type',
        hover_name='id',
        hover_data=['type', 'height_m'],
        title="Asset Locations",
        zoom=12,
        height=500
    )

    fig.update_layout(mapbox_style='open-street-map')

    return fig
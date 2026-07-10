"""
Streamlit Dashboard for GeoEnrich Core
Complete platform with OSM, LiDAR, ML, Time Series, Routing, Geocoding, Places, Elevation, and Enrichment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import sys
import os
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.real_data_loader import (
    load_amsterdam_data,
    clear_cache,
    ASSET_COLORS
)
from src.analytics.spatial_analytics import SpatialAnalytics
from src.analytics.asset_recognizer import AssetRecognizer
from src.analytics.time_series_analyzer import TimeSeriesAnalyzer
from src.ml.asset_classifier import AssetClassifier, train_asset_classifier, predict_assets

# New imports for added features
from src.core.geocoding import GeocodingService
from src.core.places_search import PlacesSearch
from src.core.routing_analyzer import RoutingAnalyzer
from src.core.elevation_service import ElevationService
from src.core.data_enrichment import DataEnrichment

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="GeoEnrich Core - Asset Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .data-source-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.1rem;
        background: #4ECDC4;
        color: white;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.assets = None
    st.session_state.point_cloud = None
    st.session_state.road_network = None
    st.session_state.analyzed = False
    st.session_state.analysis_results = None
    st.session_state.data_source = None
    st.session_state.is_real_data = False
    st.session_state.map_type = "OpenStreetMap"
    st.session_state.use_osm = True
    st.session_state.ml_trained = False
    st.session_state.timeline = None

    # New session states
    st.session_state.geocoding_result = None
    st.session_state.routing_result = None
    st.session_state.places_result = None
    st.session_state.elevation_result = None
    st.session_state.enrichment_result = None


# ============================================================================
# Initialize Services
# ============================================================================

@st.cache_resource
def get_geocoding_service():
    return GeocodingService()


@st.cache_resource
def get_places_service():
    return PlacesSearch()


@st.cache_resource
def get_routing_service():
    return RoutingAnalyzer()


@st.cache_resource
def get_elevation_service():
    return ElevationService()


@st.cache_resource
def get_enrichment_service():
    return DataEnrichment()


# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_data
def load_sample_data():
    """Load sample data."""
    try:
        assets = pd.read_csv('outputs/sample_data/assets.csv')
        road = pd.read_csv('outputs/sample_data/road_network.csv')
        point_cloud = np.load('outputs/sample_data/point_cloud.npy', allow_pickle=True)
        return assets, road, point_cloud
    except:
        import os
        os.makedirs('outputs/sample_data', exist_ok=True)
        from src.core.data_generation import MobileMappingGenerator
        generator = MobileMappingGenerator()
        files = generator.generate_full_dataset('outputs/sample_data')
        return load_sample_data()


def load_real_assets_only(max_assets=500, use_osm=True):
    """Load real assets with OSM integration."""
    try:
        assets, road, pc = load_amsterdam_data(
            use_real=True,
            max_assets=max_assets,
            use_osm=use_osm
        )
        if assets is not None and len(assets) > 0:
            return assets, road, pc
        else:
            return None, None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None


@st.cache_data
def run_analysis(assets, point_cloud):
    """Run spatial analysis and asset recognition."""
    proximity = SpatialAnalytics.calculate_proximity(assets)
    clusters = SpatialAnalytics.cluster_assets(assets)
    distribution = SpatialAnalytics.analyze_distribution(assets)

    recognizer = AssetRecognizer()
    recognized = recognizer.recognize_assets(assets, point_cloud)
    recognition_df = recognizer.generate_report()

    return {
        'proximity': proximity,
        'clusters': clusters,
        'distribution': distribution,
        'recognition': recognition_df
    }


def train_ml_model(assets):
    """Train ML model on assets."""
    if len(assets) < 10:
        return None

    classifier = AssetClassifier()
    results = classifier.train(assets)
    return results


def generate_timeline(assets):
    """Generate time series data."""
    analyzer = TimeSeriesAnalyzer()
    timeline = analyzer.generate_timeline(assets, n_steps=8, interval_days=30)
    return timeline, analyzer


def create_base_map(center_lat, center_lon, map_type="OpenStreetMap", zoom=14):
    """Create a folium map."""
    if map_type == "Satellite":
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        attr = 'Tiles &copy; Esri'
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=tiles, attr=attr)
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
    return m


# ============================================================================
# New Feature Functions
# ============================================================================

def handle_geocoding(geocoder, address):
    """Handle geocoding request."""
    if not address:
        return None
    result = geocoder.geocode_address(address)
    return result


def handle_places_search(places_service, lat, lon, category, radius):
    """Handle places search request."""
    if lat is None or lon is None:
        return None
    df = places_service.search_nearby(lat, lon, radius_km=radius, category=category)
    return df


def handle_routing(routing_service, origin, destination):
    """Handle routing request."""
    if origin is None or destination is None:
        return None
    result = routing_service.find_shortest_route(origin, destination)
    return result


def handle_elevation(elevation_service, lat, lon):
    """Handle elevation request."""
    if lat is None or lon is None:
        return None
    elevation = elevation_service.get_elevation(lat, lon)
    return elevation


def handle_enrichment(enrichment_service, assets_df):
    """Handle data enrichment request."""
    if assets_df is None or len(assets_df) == 0:
        return None
    result = enrichment_service.enrich_assets_with_demographics(assets_df)
    return result


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    """Main dashboard function."""

    # Header
    st.markdown('<div class="main-header">🌍 GeoEnrich Core</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Geospatial Intelligence & Asset Enrichment Engine</div>',
                unsafe_allow_html=True)

    # ============================================================================
    # Sidebar
    # ============================================================================

    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; padding: 0.5rem 0;">
            <div style="
                width: 90px;
                height: 90px;
                background: radial-gradient(ellipse at 30% 30%, #1a365d, #0d1b2a);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto;
                border: 2px solid #4ECDC4;
            ">
                <svg width="50" height="50" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="10" stroke="#4ECDC4" stroke-width="1.5"/>
                    <ellipse cx="12" cy="12" rx="10" ry="4" stroke="#4ECDC4" stroke-width="1" stroke-dasharray="2 2" opacity="0.5"/>
                    <circle cx="12" cy="12" r="3" fill="#4ECDC4" opacity="0.3"/>
                    <circle cx="12" cy="12" r="1.5" fill="#4ECDC4"/>
                    <circle cx="7" cy="7" r="1.5" fill="#FF6B6B" opacity="0.7"/>
                    <circle cx="17" cy="7" r="1.5" fill="#FFD93D" opacity="0.7"/>
                    <circle cx="17" cy="17" r="1.5" fill="#FF6B6B" opacity="0.7"/>
                    <circle cx="7" cy="17" r="1.5" fill="#FFD93D" opacity="0.7"/>
                    <line x1="7" y1="7" x2="12" y2="12" stroke="#4ECDC4" stroke-width="0.8" opacity="0.5"/>
                    <line x1="12" y1="12" x2="17" y2="7" stroke="#4ECDC4" stroke-width="0.8" opacity="0.5"/>
                    <line x1="12" y1="12" x2="17" y2="17" stroke="#4ECDC4" stroke-width="0.8" opacity="0.5"/>
                    <line x1="12" y1="12" x2="7" y2="17" stroke="#4ECDC4" stroke-width="0.8" opacity="0.5"/>
                </svg>
            </div>
            <div style="margin-top: 0.5rem; text-align: center;">
                <h3 style="margin: 0; color: #1a365d; font-weight: 700; font-size: 1.2rem;">
                    Geo<span style="color: #4ECDC4;">Enrich</span>
                </h3>
                <p style="margin: 0; color: #4a5568; font-size: 0.65rem; letter-spacing: 3px;">CORE</p>
                <div style="width: 40px; height: 2px; background: linear-gradient(90deg, #4ECDC4, #FF6B6B); margin: 0.25rem auto;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.header("📊 Controls")

        # Data Source
        data_source = st.radio(
            "🌍 Data Source",
            ["🏙️ Real Amsterdam", "📊 Sample Data"],
            index=0
        )

        # OSM Toggle
        use_osm = st.checkbox("🌐 Use OpenStreetMap API", value=st.session_state.use_osm)
        st.session_state.use_osm = use_osm

        max_assets = st.slider(
            "Max Assets",
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )

        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading data..."):
                if data_source == "🏙️ Real Amsterdam":
                    try:
                        assets, road, pc = load_real_assets_only(
                            max_assets=max_assets,
                            use_osm=st.session_state.use_osm
                        )
                        if assets is not None and len(assets) > 0:
                            st.session_state.data_source = f"Amsterdam ({len(assets)} assets)"
                            st.session_state.is_real_data = True
                        else:
                            assets, road, pc = load_sample_data()
                            st.session_state.data_source = "Sample (fallback)"
                            st.session_state.is_real_data = False
                    except Exception as e:
                        st.warning(f"Error: {str(e)}. Using sample data.")
                        assets, road, pc = load_sample_data()
                        st.session_state.data_source = "Sample (fallback)"
                        st.session_state.is_real_data = False
                else:
                    assets, road, pc = load_sample_data()
                    st.session_state.data_source = "Sample Data"
                    st.session_state.is_real_data = False

                st.session_state.assets = assets
                st.session_state.road_network = road
                st.session_state.point_cloud = pc
                st.session_state.data_loaded = True
                st.session_state.analyzed = False
                st.session_state.ml_trained = False
                st.session_state.timeline = None

                st.success(f"✅ Loaded {len(assets)} assets")
                st.rerun()

        if st.button("🗑️ Clear Cache"):
            clear_cache()
            st.success("✅ Cache cleared!")

        st.markdown("---")

        if st.session_state.data_loaded:
            st.markdown("### 📁 Data Summary")
            st.metric("Assets", len(st.session_state.assets))
            st.metric("Point Cloud", f"{len(st.session_state.point_cloud):,} points")
            st.metric("Asset Types", len(st.session_state.assets['type'].unique()))

            if st.session_state.is_real_data:
                st.markdown(f'<span class="data-source-badge">✅ Amsterdam Data</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="data-source-badge">📊 Sample Data</span>', unsafe_allow_html=True)

            if st.session_state.analyzed and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                dist = results['distribution']
                st.metric("Mean Distance", f"{dist.get('mean_distance_km', 0):.3f} km")

        st.markdown("---")
        st.caption("GeoEnrich Core v2.0 | All Features Integrated")

    # ============================================================================
    # Main Content
    # ============================================================================

    if not st.session_state.data_loaded:
        st.info("👈 Load data to get started")
        return

    # Tabs - Added new tabs for features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🗺️ Map View",
        "📊 Analytics",
        "🤖 Asset Recognition",
        "🧠 ML Classification",
        "📈 Time Series",
        "📍 Geocoding & Places",
        "🗺️ Routing & Elevation",
        "📋 Reports"
    ])

    # ========================================================================
    # TAB 1: Map View
    # ========================================================================
    with tab1:
        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            st.subheader("Interactive Asset Map")

            map_type = st.radio(
                "🗺️ Map Layer",
                ["OpenStreetMap", "Satellite"],
                index=0,
                horizontal=True
            )
            st.session_state.map_type = map_type

            center_lat = st.session_state.assets['lat'].mean()
            center_lon = st.session_state.assets['lon'].mean()

            m = create_base_map(center_lat, center_lon, map_type=map_type, zoom=14)

            # Road network
            if st.session_state.road_network is not None and len(st.session_state.road_network) > 0:
                road_points = [[row['lat'], row['lon']]
                               for _, row in st.session_state.road_network.iterrows()]
                folium.PolyLine(road_points, color='#FF6B6B', weight=2, opacity=0.7).add_to(m)

            # Assets
            assets_to_plot = st.session_state.assets
            if len(assets_to_plot) > 1000:
                assets_to_plot = assets_to_plot.sample(1000)

            for _, asset in assets_to_plot.iterrows():
                color = ASSET_COLORS.get(asset['type'], '#95A5A6')
                height = asset.get('height_m', 0)

                popup = f"""
                <div style="font-family: Arial, sans-serif; padding: 4px;">
                    <b style="color: {color};">{asset['type']}</b><br>
                    ID: {asset['id']}<br>
                    Height: {height:.2f}m
                    Source: {asset.get('source', 'unknown')}
                </div>
                """

                folium.CircleMarker(
                    location=[asset['lat'], asset['lon']],
                    radius=5,
                    popup=popup,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8
                ).add_to(m)

            folium_static(m, width=700, height=500)

        with col2:
            st.subheader("Asset Summary")

            if st.session_state.is_real_data:
                st.markdown(f'<span class="data-source-badge">✅ Amsterdam Data</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="data-source-badge">📊 Sample Data</span>', unsafe_allow_html=True)

            st.markdown("---")

            counts = st.session_state.assets['type'].value_counts()
            for asset_type, count in counts.head(10).items():
                color = ASSET_COLORS.get(asset_type, '#95A5A6')
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; 
                         padding: 0.3rem; border-bottom: 1px solid #eee;">
                        <span><span style="color:{color}; font-size:1.2rem;">●</span> {asset_type}</span>
                        <span style="font-weight:bold;">{count}</span>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.metric("Total Assets", len(st.session_state.assets))
            st.metric("Asset Types", len(counts))

            if st.session_state.analyzed and st.session_state.analysis_results:
                dist = st.session_state.analysis_results['distribution']
                st.metric("Mean Distance", f"{dist.get('mean_distance_km', 0):.4f} km")

    # ========================================================================
    # TAB 2: Analytics
    # ========================================================================
    with tab2:
        st.subheader("Analytics Dashboard")

        if not st.session_state.analyzed:
            if st.button("🔬 Run Analysis", type="primary"):
                with st.spinner("Running analysis..."):
                    results = run_analysis(
                        st.session_state.assets,
                        st.session_state.point_cloud
                    )
                    st.session_state.analysis_results = results
                    st.session_state.analyzed = True
                st.success("✅ Analysis complete!")
                st.rerun()

        if st.session_state.analyzed and st.session_state.analysis_results:
            results = st.session_state.analysis_results

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Asset Type Distribution")
                counts = st.session_state.assets['type'].value_counts()
                if len(counts) > 10:
                    top_counts = counts.head(10)
                    other = counts[10:].sum()
                    if other > 0:
                        top_counts['Other'] = other
                    counts = top_counts

                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    color=counts.index,
                    color_discrete_map=ASSET_COLORS,
                    hole=0.3
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Asset Heights")
                height_data = st.session_state.assets.sample(min(500, len(st.session_state.assets)))
                fig = px.bar(
                    height_data,
                    x='id',
                    y='height_m',
                    color='type',
                    color_discrete_map=ASSET_COLORS
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Proximity Analysis")
                prox = results['proximity']

                neighbor_counts = prox['neighbor_count'].value_counts().sort_index()
                fig = px.bar(
                    x=neighbor_counts.index,
                    y=neighbor_counts.values,
                    labels={'x': 'Neighbors', 'y': 'Asset Count'}
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

                isolated = prox[prox['is_isolated'] == True]
                st.metric("Isolated Assets", len(isolated))

                clusters = results['clusters']
                cluster_counts = clusters.groupby('cluster_id').size()
                st.write(f"**Clusters:** {len(cluster_counts)}")
                st.write(f"**Largest Cluster:** {cluster_counts.max()} assets")
                st.write(f"**Avg Cluster Size:** {cluster_counts.mean():.1f}")

    # ========================================================================
    # TAB 3: Asset Recognition
    # ========================================================================
    with tab3:
        st.subheader("🤖 AI Asset Recognition")

        if not st.session_state.analyzed:
            st.info("Run analysis first")
        else:
            recognition_df = st.session_state.analysis_results['recognition']

            valid = len(recognition_df[recognition_df['is_valid'] == True])
            invalid = len(recognition_df[recognition_df['is_valid'] == False])
            total = len(recognition_df)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Assets", total)
            with col2:
                st.metric("✅ Valid", valid, delta=f"{valid / total * 100:.0f}%")
            with col3:
                st.metric("❌ Invalid", invalid)

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    recognition_df,
                    x='confidence',
                    nbins=20,
                    color='is_valid',
                    color_discrete_map={True: '#27ae60', False: '#e74c3c'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                valid_by_type = recognition_df.groupby('recognized_type')['is_valid'].mean().sort_values(
                    ascending=False)
                fig = px.bar(
                    x=valid_by_type.values,
                    y=valid_by_type.index,
                    orientation='h',
                    color=valid_by_type.values,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.dataframe(recognition_df.head(100), use_container_width=True)

    # ========================================================================
    # TAB 4: ML Classification
    # ========================================================================
    with tab4:
        st.subheader("🧠 Machine Learning Classification")

        col1, col2 = st.columns([0.5, 0.5])

        with col1:
            if st.button("🎯 Train ML Model", type="primary"):
                with st.spinner("Training model..."):
                    results = train_ml_model(st.session_state.assets)
                    if results:
                        st.session_state.ml_trained = True
                        st.success(f"✅ Model trained! Accuracy: {results['accuracy'] * 100:.1f}%")
                    else:
                        st.warning("Not enough data to train model")

        with col2:
            if st.session_state.ml_trained:
                if st.button("🔮 Predict Assets"):
                    with st.spinner("Making predictions..."):
                        results = predict_assets(st.session_state.assets)
                        if results is not None:
                            st.session_state.ml_results = results
                            st.success("✅ Predictions complete!")

        if st.session_state.ml_trained:
            st.divider()
            st.markdown("#### ML Model Performance")

            if 'ml_results' in st.session_state:
                df = st.session_state.ml_results
                st.dataframe(df.head(50), use_container_width=True)

                fig = px.histogram(
                    df,
                    x='ml_confidence',
                    nbins=20,
                    title="ML Confidence Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 5: Time Series
    # ========================================================================
    with tab5:
        st.subheader("📈 Time Series Analysis")

        if st.button("📊 Generate Timeline"):
            with st.spinner("Generating timeline..."):
                timeline, analyzer = generate_timeline(st.session_state.assets)
                st.session_state.timeline = timeline
                st.session_state.timeline_analyzer = analyzer
                st.success("✅ Timeline generated!")

        if st.session_state.timeline is not None:
            timeline = st.session_state.timeline

            st.markdown("#### Asset Changes Over Time")

            counts_over_time = timeline.groupby(['step', 'type']).size().reset_index(name='count')
            fig = px.line(
                counts_over_time,
                x='step',
                y='count',
                color='type',
                title="Asset Count Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

            height_trend = timeline.groupby(['step', 'type'])['height_m'].mean().reset_index()
            fig = px.line(
                height_trend,
                x='step',
                y='height_m',
                color='type',
                title="Average Height Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Change Summary")
            steps = sorted(timeline['step'].unique())
            if len(steps) >= 2:
                snap1 = timeline[timeline['step'] == steps[0]]
                snap2 = timeline[timeline['step'] == steps[-1]]

                summary = st.session_state.timeline_analyzer.get_change_summary(snap1, snap2)
                st.text(summary)

            if st.button("📈 Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    forecast = st.session_state.timeline_analyzer.generate_forecast(
                        timeline, steps_ahead=5
                    )
                    if len(forecast) > 0:
                        st.dataframe(forecast, use_container_width=True)

                        fig = px.line(
                            forecast,
                            x='step',
                            y='forecast_height_m',
                            color='type',
                            title="Height Forecast"
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 6: Geocoding & Places - NEW
    # ========================================================================
    with tab6:
        st.subheader("📍 Geocoding & Places Search")

        # Initialize services
        geocoder = get_geocoding_service()
        places = get_places_service()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔍 Geocoding")
            address_input = st.text_input("Enter address:", placeholder="Dam Square, Amsterdam")

            if st.button("Geocode Address", type="primary"):
                if address_input:
                    with st.spinner("Geocoding..."):
                        result = handle_geocoding(geocoder, address_input)
                        if result:
                            st.session_state.geocoding_result = result
                            st.success("✅ Geocoding successful!")
                        else:
                            st.error("❌ Address not found")

            if st.session_state.geocoding_result:
                result = st.session_state.geocoding_result
                st.markdown("### Geocoding Result")
                st.json(result)

                # Show on map
                if 'lat' in result and 'lon' in result:
                    m = folium.Map(location=[result['lat'], result['lon']], zoom_start=15)
                    folium.Marker(
                        [result['lat'], result['lon']],
                        popup=result.get('full_address', 'Location'),
                        icon=folium.Icon(color='green', icon='info-sign')
                    ).add_to(m)
                    folium_static(m, width=350, height=300)

        with col2:
            st.markdown("#### 🏪 Places Search")

            # Get coordinates for search
            search_lat = st.number_input("Latitude:", value=52.3676, format="%.6f")
            search_lon = st.number_input("Longitude:", value=4.9041, format="%.6f")
            search_radius = st.slider("Radius (km):", 0.5, 5.0, 1.0, 0.5)
            place_category = st.selectbox(
                "Category:",
                ["all", "restaurant", "cafe", "pharmacy", "hospital", "school", "park", "museum", "hotel", "bank",
                 "supermarket"]
            )

            if st.button("Search Places", type="primary"):
                with st.spinner("Searching places..."):
                    category = None if place_category == "all" else place_category
                    result = handle_places_search(places, search_lat, search_lon, category, search_radius)
                    if result is not None and len(result) > 0:
                        st.session_state.places_result = result
                        st.success(f"✅ Found {len(result)} places")
                    else:
                        st.warning("No places found")

            if st.session_state.places_result is not None and len(st.session_state.places_result) > 0:
                df = st.session_state.places_result
                st.dataframe(df, use_container_width=True)

    # ========================================================================
    # TAB 7: Routing & Elevation - NEW
    # ========================================================================
    with tab7:
        st.subheader("🗺️ Routing & Elevation")

        routing = get_routing_service()
        elevation = get_elevation_service()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🚗 Routing")

            st.markdown("**Origin:**")
            origin_lat = st.number_input("Origin Lat:", value=52.3676, key="origin_lat", format="%.6f")
            origin_lon = st.number_input("Origin Lon:", value=4.9041, key="origin_lon", format="%.6f")

            st.markdown("**Destination:**")
            dest_lat = st.number_input("Dest Lat:", value=52.3700, key="dest_lat", format="%.6f")
            dest_lon = st.number_input("Dest Lon:", value=4.9100, key="dest_lon", format="%.6f")

            if st.button("Find Route", type="primary"):
                with st.spinner("Finding route..."):
                    result = handle_routing(
                        routing,
                        (origin_lat, origin_lon),
                        (dest_lat, dest_lon)
                    )
                    if result and 'route_points' in result and len(result['route_points']) > 0:
                        st.session_state.routing_result = result
                        st.success(f"✅ Route found! Distance: {result['total_distance_km']:.3f} km")
                    else:
                        st.error("❌ No route found")

            if st.session_state.routing_result:
                result = st.session_state.routing_result
                st.markdown("### Route Details")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Distance", f"{result.get('total_distance_km', 0):.3f} km")
                with col_b:
                    st.metric("Est. Time", f"{result.get('estimated_time_min', 0):.1f} min")

                # Show route on map
                if 'route_points' in result and len(result['route_points']) > 0:
                    route_points = result['route_points']
                    center_lat = route_points[0]['lat'] if route_points else 52.3676
                    center_lon = route_points[0]['lon'] if route_points else 4.9041

                    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

                    # Add route line
                    coords = [[p['lat'], p['lon']] for p in route_points]
                    folium.PolyLine(coords, color='blue', weight=3, opacity=0.8, popup='Route').add_to(m)

                    # Add markers
                    folium.Marker(
                        [origin_lat, origin_lon],
                        popup='Origin',
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(m)
                    folium.Marker(
                        [dest_lat, dest_lon],
                        popup='Destination',
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(m)

                    folium_static(m, width=400, height=300)

        with col2:
            st.markdown("#### ⛰️ Elevation")

            elev_lat = st.number_input("Elevation Lat:", value=52.3676, key="elev_lat", format="%.6f")
            elev_lon = st.number_input("Elevation Lon:", value=4.9041, key="elev_lon", format="%.6f")

            if st.button("Get Elevation", type="primary"):
                with st.spinner("Getting elevation..."):
                    elev = handle_elevation(elevation, elev_lat, elev_lon)
                    if elev is not None:
                        st.session_state.elevation_result = elev
                        st.success(f"✅ Elevation: {elev:.2f}m")
                    else:
                        st.error("❌ Elevation unavailable")

            if st.session_state.elevation_result is not None:
                st.metric("Elevation", f"{st.session_state.elevation_result:.2f} m")

                # Show elevation profile
                st.markdown("#### Elevation Profile")
                # Generate sample profile
                profile_points = [
                    {'lat': elev_lat - 0.001, 'lon': elev_lon},
                    {'lat': elev_lat, 'lon': elev_lon},
                    {'lat': elev_lat + 0.001, 'lon': elev_lon}
                ]
                profile = elevation.get_elevation_profile(profile_points)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(profile['elevations']))),
                    y=profile['elevations'],
                    mode='lines+markers',
                    name='Elevation Profile',
                    line=dict(color='#4ECDC4', width=3)
                ))
                fig.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    yaxis_title="Elevation (m)",
                    xaxis_title="Point"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 8: Reports
    # ========================================================================
    with tab8:
        st.subheader("📋 Reports & Exports")

        if st.session_state.analyzed:
            results = st.session_state.analysis_results
            assets = st.session_state.assets

            st.markdown("### Executive Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Assets", len(assets))
            with col2:
                st.metric("Asset Types", len(assets['type'].unique()))
            with col3:
                valid = len(results['recognition'][results['recognition']['is_valid'] == True])
                st.metric("✅ Valid Assets", valid)
            with col4:
                clusters = len(results['clusters']['cluster_id'].unique())
                st.metric("Clusters", clusters)

            st.divider()

            # Data Enrichment
            st.markdown("### 📊 Data Enrichment")
            if st.button("Enrich Asset Data", type="primary"):
                with st.spinner("Enriching data..."):
                    enrichment = get_enrichment_service()
                    result = handle_enrichment(enrichment, st.session_state.assets)
                    if result is not None:
                        st.session_state.enrichment_result = result
                        st.success("✅ Data enriched!")

            if st.session_state.enrichment_result is not None:
                st.dataframe(st.session_state.enrichment_result.head(100), use_container_width=True)

            st.divider()

            st.markdown("### Export Data")

            col1, col2, col3 = st.columns(3)

            with col1:
                csv_data = assets.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="assets_export.csv",
                    mime="text/csv"
                )

            with col2:
                import json
                geojson = {
                    "type": "FeatureCollection",
                    "features": []
                }
                for _, asset in assets.iterrows():
                    geojson["features"].append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [asset['lon'], asset['lat']]
                        },
                        "properties": {
                            "id": asset['id'],
                            "type": asset['type'],
                            "height_m": asset.get('height_m', 0),
                            "source": asset.get('source', 'unknown')
                        }
                    })
                st.download_button(
                    label="🗺️ Download GeoJSON",
                    data=json.dumps(geojson, indent=2),
                    file_name="assets.geojson",
                    mime="application/json"
                )

            with col3:
                rec_data = results['recognition'].to_csv(index=False)
                st.download_button(
                    label="🤖 Recognition Report",
                    data=rec_data,
                    file_name="recognition_report.csv",
                    mime="text/csv"
                )

            st.divider()

            st.markdown("### Data Tables")
            tab_a, tab_b = st.tabs(["Assets", "Recognition"])

            with tab_a:
                st.dataframe(assets.head(500), use_container_width=True)

            with tab_b:
                st.dataframe(results['recognition'].head(500), use_container_width=True)
        else:
            st.info("Run analysis to see reports.")


if __name__ == "__main__":
    main()
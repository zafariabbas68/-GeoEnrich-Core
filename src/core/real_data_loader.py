"""
Real Data Loader - OSM Integration + Amsterdam Data
Fetches real data from OpenStreetMap with caching.
"""

import os
import pandas as pd
import numpy as np
import random
import json
import requests
from datetime import datetime
import time

# ============================================================================
# Asset Colors
# ============================================================================

ASSET_COLORS = {
    'Streetlight': '#FF6B6B',
    'Traffic Sign': '#4ECDC4',
    'Tree': '#45B7D1',
    'Manhole': '#96CEB4',
    'Fire Hydrant': '#FFEAA7',
    'Building Corner': '#DDA0DD',
    'Vehicle': '#FF9FF3',
    'Bench': '#FF9F43',
    'Bike Rack': '#00D2D3',
    'Bus Stop': '#54A0FF',
    'Telephone': '#5F27CD',
    'Post Box': '#FF6B81',
    'Waste Basket': '#8395A7',
    'Drinking Water': '#1DD1A1',
    'Toilet': '#F368E0',
    'Unknown': '#95A5A6'
}

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
OSM_CACHE_FILE = os.path.join(CACHE_DIR, 'osm_amsterdam_data.pkl')
MOCK_CACHE_FILE = os.path.join(CACHE_DIR, 'amsterdam_assets_accurate.pkl')


class OSMDataFetcher:
    """
    Fetch real data from OpenStreetMap Overpass API.
    """

    # Overpass API endpoint
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    # OSM tags for each asset type
    OSM_TAGS = {
        'Streetlight': {'highway': 'street_lamp'},
        'Traffic Sign': {'traffic_sign': True},
        'Tree': {'natural': 'tree'},
        'Fire Hydrant': {'emergency': 'fire_hydrant'},
        'Bench': {'amenity': 'bench'},
        'Bike Rack': {'amenity': 'bicycle_parking'},
        'Bus Stop': {'highway': 'bus_stop'},
        'Post Box': {'amenity': 'post_box'},
        'Waste Basket': {'amenity': 'waste_basket'},
        'Drinking Water': {'amenity': 'drinking_water'},
        'Toilet': {'amenity': 'toilets'},
        'Telephone': {'amenity': 'telephone'},
    }

    # Amsterdam bounding box
    AMSTERDAM_BBOX = {
        'min_lat': 52.30,
        'max_lat': 52.42,
        'min_lon': 4.80,
        'max_lon': 4.98
    }

    def __init__(self):
        self.cache = {}
        self.fetched_data = []

    def fetch_osm_data(self, asset_type, bbox=None, max_results=100):
        """
        Fetch data from OpenStreetMap Overpass API.

        Args:
            asset_type: Type of asset to fetch
            bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
            max_results: Maximum results to return

        Returns:
            List of features
        """
        if bbox is None:
            bbox = self.AMSTERDAM_BBOX

        # Check cache first
        cache_key = f"{asset_type}_{bbox['min_lat']}_{bbox['min_lon']}_{bbox['max_lat']}_{bbox['max_lon']}"
        if cache_key in self.cache:
            print(f"📦 Using cached data for {asset_type}")
            return self.cache[cache_key][:max_results]

        tags = self.OSM_TAGS.get(asset_type)
        if tags is None:
            print(f"⚠️ No OSM tags for {asset_type}")
            return []

        # Build query
        query = self._build_query(asset_type, tags, bbox)

        try:
            print(f"🌍 Fetching {asset_type} from OSM...")
            response = requests.post(self.OVERPASS_URL, data={'data': query}, timeout=30)
            response.raise_for_status()

            data = response.json()
            features = data.get('elements', [])

            # Cache the results
            self.cache[cache_key] = features
            print(f"✅ Fetched {len(features)} {asset_type}(s)")

            return features[:max_results]

        except requests.exceptions.RequestException as e:
            print(f"⚠️ OSM fetch failed for {asset_type}: {str(e)}")
            return []

    def _build_query(self, asset_type, tags, bbox):
        """Build Overpass API query."""
        tag_str = ' '.join([f'["{k}"="{v}"]' for k, v in tags.items()])

        query = f"""
        [out:json][timeout:30];
        (
          node{tag_str}({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
          way{tag_str}({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
        );
        out body;
        """
        return query.strip()

    def fetch_all_assets(self, max_assets=500):
        """
        Fetch all asset types from OSM.

        Args:
            max_assets: Maximum total assets

        Returns:
            DataFrame with assets
        """
        all_assets = []

        for asset_type in self.OSM_TAGS.keys():
            if len(all_assets) >= max_assets:
                break

            features = self.fetch_osm_data(asset_type, max_results=50)

            for feature in features:
                if len(all_assets) >= max_assets:
                    break

                # Extract coordinates
                lat = feature.get('lat')
                lon = feature.get('lon')

                if lat is None or lon is None:
                    continue

                # Get height if available
                tags = feature.get('tags', {})
                height = tags.get('height', 0)
                try:
                    height = float(height)
                except:
                    height = random.uniform(
                        self._get_height_range(asset_type)[0],
                        self._get_height_range(asset_type)[1]
                    )

                all_assets.append({
                    'id': f"{asset_type[:3]}_{len(all_assets) + 1:04d}",
                    'type': asset_type,
                    'lat': lat,
                    'lon': lon,
                    'height_m': height,
                    'confidence': 0.95,
                    'status': 'osm',
                    'source': 'OSM',
                    'osm_id': feature.get('id')
                })

        if len(all_assets) == 0:
            print("⚠️ No OSM data fetched. Using mock data.")
            return self._generate_mock_data(max_assets)

        print(f"✅ Total OSM assets: {len(all_assets)}")
        return pd.DataFrame(all_assets)

    def _get_height_range(self, asset_type):
        """Get height range for asset type."""
        ranges = {
            'Streetlight': (3.0, 7.0),
            'Traffic Sign': (1.0, 3.0),
            'Tree': (4.0, 15.0),
            'Fire Hydrant': (0.5, 1.0),
            'Bench': (0.4, 0.8),
            'Bike Rack': (0.5, 1.0),
            'Bus Stop': (2.0, 3.0),
            'Post Box': (0.8, 1.3),
            'Waste Basket': (0.5, 0.9),
            'Drinking Water': (0.8, 1.2),
            'Toilet': (2.0, 3.0),
            'Telephone': (1.0, 2.0),
        }
        return ranges.get(asset_type, (0.5, 2.0))

    def _generate_mock_data(self, max_assets=500):
        """Generate mock data as fallback."""
        print("🏙️ Generating mock Amsterdam data...")
        center_lat, center_lon = 52.3676, 4.9041

        asset_defs = {
            'Streetlight': {'count': 45, 'height_min': 3.0, 'height_max': 7.0},
            'Traffic Sign': {'count': 30, 'height_min': 1.0, 'height_max': 3.0},
            'Tree': {'count': 50, 'height_min': 4.0, 'height_max': 15.0},
            'Manhole': {'count': 15, 'height_min': 0.0, 'height_max': 0.3},
            'Fire Hydrant': {'count': 12, 'height_min': 0.5, 'height_max': 1.0},
            'Building Corner': {'count': 18, 'height_min': 5.0, 'height_max': 25.0},
            'Vehicle': {'count': 15, 'height_min': 1.2, 'height_max': 2.5},
            'Bench': {'count': 12, 'height_min': 0.4, 'height_max': 0.8},
            'Bike Rack': {'count': 10, 'height_min': 0.5, 'height_max': 1.0},
            'Bus Stop': {'count': 8, 'height_min': 2.0, 'height_max': 3.0},
            'Post Box': {'count': 6, 'height_min': 0.8, 'height_max': 1.3},
            'Waste Basket': {'count': 6, 'height_min': 0.5, 'height_max': 0.9},
        }

        assets = []
        for asset_type, config in asset_defs.items():
            count = config['count']
            for i in range(count):
                lat = center_lat + random.uniform(-0.015, 0.015)
                lon = center_lon + random.uniform(-0.015, 0.015)
                height = random.uniform(config['height_min'], config['height_max'])
                assets.append({
                    'id': f"{asset_type[:3]}_{i + 1:04d}",
                    'type': asset_type,
                    'lat': lat,
                    'lon': lon,
                    'height_m': round(height, 2),
                    'confidence': 0.95,
                    'status': 'mock',
                    'source': 'mock'
                })

        return pd.DataFrame(assets)


def load_amsterdam_data(use_real=True, max_assets=500, use_osm=True):
    """
    Load Amsterdam data with OSM integration.

    Args:
        use_real: Whether to use real data
        max_assets: Maximum assets to return
        use_osm: Whether to use OSM API

    Returns:
        Tuple of (assets_df, road_df, point_cloud)
    """

    # Try cache first
    cache_file = OSM_CACHE_FILE if use_osm else MOCK_CACHE_FILE

    if os.path.exists(cache_file):
        try:
            assets = pd.read_pickle(cache_file)
            print(f"📦 Loaded {len(assets)} assets from cache")
            if len(assets) > max_assets:
                assets = assets.sample(max_assets)
            return assets, _generate_road_points(), _generate_point_cloud(assets)
        except Exception as e:
            print(f"⚠️ Cache load failed: {str(e)}")

    # Fetch from OSM if requested
    if use_osm and use_real:
        fetcher = OSMDataFetcher()
        assets = fetcher.fetch_all_assets(max_assets=max_assets)
    else:
        # Generate mock data
        fetcher = OSMDataFetcher()
        assets = fetcher._generate_mock_data(max_assets)

    # Cache the data
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        assets.to_pickle(cache_file)
        print(f"💾 Cached {len(assets)} assets")
    except:
        pass

    return assets, _generate_road_points(), _generate_point_cloud(assets)


def _generate_road_points():
    """Generate road network."""
    center_lat, center_lon = 52.3676, 4.9041
    road_points = []
    for i in range(300):
        t = i / 300
        if i % 25 < 12:
            lat = center_lat + (i / 300 - 0.5) * 0.04
            lon = center_lon + (t * 0.04 - 0.02) + 0.003 * np.sin(t * 7)
        else:
            lat = center_lat + (t * 0.04 - 0.02) + 0.003 * np.cos(t * 5)
            lon = center_lon + (i / 300 - 0.5) * 0.04
        road_points.append({'lat': lat, 'lon': lon})
    return pd.DataFrame(road_points)


def _generate_point_cloud(assets):
    """Generate point cloud."""
    point_cloud = []
    for _, asset in assets.iterrows():
        num_points = random.randint(15, 40)
        for _ in range(num_points):
            lat = asset['lat'] + np.random.normal(0, 0.00006)
            lon = asset['lon'] + np.random.normal(0, 0.00006)
            z = np.random.uniform(0, asset['height_m'] * 0.85)
            point_cloud.append([lat, lon, z, asset['id'], asset['type']])
    return np.array(point_cloud, dtype=object)


def clear_cache():
    """Clear cache."""
    for f in [OSM_CACHE_FILE, MOCK_CACHE_FILE]:
        if os.path.exists(f):
            os.remove(f)
    print("🗑️ Cache cleared")


OSMNX_AVAILABLE = False
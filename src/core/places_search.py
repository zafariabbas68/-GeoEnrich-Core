"""
Places Search Module
Search for points of interest and places.
"""

import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Tuple
import json


class PlacesSearch:
    """
    Search for places and points of interest.
    """

    def __init__(self):
        self.cache = {}
        self.user_agent = "GeoEnrichCore/1.0"

        # Place categories and their OSM tags
        self.categories = {
            'restaurant': {'amenity': 'restaurant'},
            'cafe': {'amenity': 'cafe'},
            'pharmacy': {'amenity': 'pharmacy'},
            'hospital': {'amenity': 'hospital'},
            'school': {'amenity': 'school'},
            'university': {'amenity': 'university'},
            'park': {'leisure': 'park'},
            'museum': {'tourism': 'museum'},
            'hotel': {'tourism': 'hotel'},
            'bank': {'amenity': 'bank'},
            'supermarket': {'shop': 'supermarket'},
            'gas_station': {'amenity': 'fuel'},
            'parking': {'amenity': 'parking'},
            'post_office': {'amenity': 'post_office'},
            'police': {'amenity': 'police'},
            'fire_station': {'amenity': 'fire_station'},
            'library': {'amenity': 'library'},
            'theatre': {'amenity': 'theatre'},
            'cinema': {'amenity': 'cinema'},
            'sports_centre': {'leisure': 'sports_centre'},
            'gym': {'leisure': 'fitness_centre'},
            'swimming_pool': {'leisure': 'swimming_pool'},
            'playground': {'leisure': 'playground'},
            'zoo': {'tourism': 'zoo'},
            'aquarium': {'tourism': 'aquarium'}
        }

    def search_nearby(self, lat: float, lon: float, radius_km: float = 1,
                      category: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """
        Search for places near a location.

        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            category: Category filter (e.g., 'restaurant')
            limit: Maximum results

        Returns:
            DataFrame with place information
        """
        # Convert radius to degrees (approximate)
        radius_deg = radius_km / 111.32

        bbox = {
            'min_lat': lat - radius_deg,
            'max_lat': lat + radius_deg,
            'min_lon': lon - radius_deg,
            'max_lon': lon + radius_deg
        }

        return self.search_in_bbox(bbox, category=category, limit=limit)

    def search_in_bbox(self, bbox: Dict[str, float],
                       category: Optional[str] = None,
                       limit: int = 50) -> pd.DataFrame:
        """
        Search for places within a bounding box.

        Args:
            bbox: {'min_lat', 'max_lat', 'min_lon', 'max_lon'}
            category: Category filter
            limit: Maximum results

        Returns:
            DataFrame with place information
        """
        # Build query
        if category and category in self.categories:
            tags = self.categories[category]
            tag_str = ' '.join([f'["{k}"="{v}"]' for k, v in tags.items()])
        else:
            # Search all categories
            tag_str = ''

        query = f"""
        [out:json][timeout:30];
        (
          node{tag_str}({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
          way{tag_str}({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
        );
        out body;
        """

        try:
            url = "https://overpass-api.de/api/interpreter"
            response = requests.post(url, data={'data': query}, timeout=30)
            response.raise_for_status()

            data = response.json()
            features = data.get('elements', [])

            places = []
            for feature in features[:limit]:
                lat = feature.get('lat')
                lon = feature.get('lon')
                if lat is None or lon is None:
                    continue

                tags = feature.get('tags', {})
                name = tags.get('name', 'Unknown')

                # Determine category from tags
                place_category = 'unknown'
                for cat, cat_tags in self.categories.items():
                    if all(tags.get(k) == v for k, v in cat_tags.items()):
                        place_category = cat
                        break

                places.append({
                    'id': f"place_{feature.get('id', len(places))}",
                    'name': name,
                    'category': place_category,
                    'lat': lat,
                    'lon': lon,
                    'address': tags.get('addr:street', ''),
                    'phone': tags.get('phone', ''),
                    'website': tags.get('website', ''),
                    'osm_id': feature.get('id')
                })

            return pd.DataFrame(places)

        except Exception as e:
            print(f"⚠️ Places search error: {str(e)}")
            return pd.DataFrame()

    def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a place.

        Args:
            place_id: Place identifier

        Returns:
            Dictionary with place details
        """
        # Implementation would fetch from OSM or another API
        # For demo, return mock data
        return {
            'id': place_id,
            'name': 'Sample Place',
            'category': 'restaurant',
            'rating': 4.5,
            'review_count': 120,
            'phone': '+31 20 123 4567',
            'website': 'https://example.com',
            'hours': 'Mon-Sun: 09:00-22:00',
            'price_range': '$$',
            'description': 'A sample place description'
        }

    def search_by_name(self, name: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for places by name.

        Args:
            name: Place name or partial name
            limit: Maximum results

        Returns:
            DataFrame with place information
        """
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': name,
                'format': 'json',
                'limit': limit,
                'category': 'amenity',
                'addressdetails': 1
            }
            headers = {'User-Agent': self.user_agent}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            places = []
            for item in data:
                places.append({
                    'id': f"search_{len(places)}",
                    'name': item.get('display_name', 'Unknown'),
                    'category': item.get('type', 'unknown'),
                    'lat': float(item.get('lat', 0)),
                    'lon': float(item.get('lon', 0)),
                    'address': item.get('address', {})
                })

            return pd.DataFrame(places)

        except Exception as e:
            print(f"⚠️ Search by name error: {str(e)}")
            return pd.DataFrame()
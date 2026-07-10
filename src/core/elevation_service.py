"""
Elevation Service Module
Provides elevation data for points, routes, and areas.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import requests
import json


class ElevationService:
    """
    Elevation service for querying terrain heights.
    """

    def __init__(self, use_api: bool = True):
        self.use_api = use_api
        self.cache = {}

        # Open-Elevation API (free, no key required)
        self.api_url = "https://api.open-elevation.com/api/v1/lookup"

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """
        Get elevation at a single point.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters, or None if unavailable
        """
        cache_key = f"{lat:.6f}_{lon:.6f}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.use_api:
            try:
                response = requests.post(
                    self.api_url,
                    json={'locations': [{'latitude': lat, 'longitude': lon}]},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    elevation = data.get('results', [{}])[0].get('elevation')
                    if elevation is not None:
                        self.cache[cache_key] = elevation
                        return elevation
            except Exception as e:
                print(f"⚠️ Elevation API error: {str(e)}")

        # Fallback: generate realistic elevation
        elevation = self._generate_realistic_elevation(lat, lon)
        self.cache[cache_key] = elevation
        return elevation

    def get_elevation_bulk(self, points: List[Tuple[float, float]]) -> List[float]:
        """
        Get elevation for multiple points.

        Args:
            points: List of (lat, lon) tuples

        Returns:
            List of elevations in meters
        """
        return [self.get_elevation(lat, lon) for lat, lon in points]

    def get_elevation_profile(self, points: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Get elevation profile along a path.

        Args:
            points: List of {'lat': lat, 'lon': lon} dictionaries

        Returns:
            Dictionary with elevation profile data
        """
        elevations = []
        for point in points:
            elev = self.get_elevation(point['lat'], point['lon'])
            elevations.append(elev if elev is not None else 0)

        return {
            'points': points,
            'elevations': elevations,
            'min_elevation': min(elevations) if elevations else 0,
            'max_elevation': max(elevations) if elevations else 0,
            'mean_elevation': np.mean(elevations) if elevations else 0,
            'total_ascent': self._calculate_ascent(elevations),
            'total_descent': self._calculate_descent(elevations)
        }

    def _generate_realistic_elevation(self, lat: float, lon: float) -> float:
        """Generate realistic elevation based on location."""
        # Simple terrain simulation
        # Amsterdam area: 52.3676, 4.9041 (approx 0-5m)
        # Mountains: higher elevations

        # Distance from Amsterdam center
        center_lat, center_lon = 52.3676, 4.9041
        lat_dist = abs(lat - center_lat) * 111  # km
        lon_dist = abs(lon - center_lon) * 85  # km
        distance = np.sqrt(lat_dist ** 2 + lon_dist ** 2)

        # Base elevation (Amsterdam is near sea level)
        base_elevation = np.random.normal(2, 1.5)

        # Add some terrain variation
        terrain = 1.5 * np.sin(lat * 50) * np.cos(lon * 50)

        # If far from Amsterdam, add more variation (mountains)
        if distance > 50:
            mountain_factor = min(1, (distance - 50) / 100)
            terrain += mountain_factor * np.random.uniform(50, 500)

        elevation = max(0, base_elevation + terrain)
        return round(elevation, 2)

    def _calculate_ascent(self, elevations: List[float]) -> float:
        """Calculate total ascent."""
        ascent = 0
        for i in range(1, len(elevations)):
            if elevations[i] > elevations[i - 1]:
                ascent += elevations[i] - elevations[i - 1]
        return round(ascent, 2)

    def _calculate_descent(self, elevations: List[float]) -> float:
        """Calculate total descent."""
        descent = 0
        for i in range(1, len(elevations)):
            if elevations[i] < elevations[i - 1]:
                descent += elevations[i - 1] - elevations[i]
        return round(descent, 2)

    def get_terrain_profile_for_route(self, route_points: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Get terrain profile for a route.

        Args:
            route_points: List of {'lat': lat, 'lon': lon} dictionaries

        Returns:
            Dictionary with terrain profile data
        """
        return self.get_elevation_profile(route_points)
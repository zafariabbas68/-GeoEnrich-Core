"""
Geocoding Module
Provides forward and reverse geocoding capabilities.
"""

import pandas as pd
import requests
from typing import Optional, Dict, Any, List, Tuple


class GeocodingService:
    """
    Geocoding service for converting addresses to coordinates and vice versa.
    """

    def __init__(self):
        self.cache = {}
        self.user_agent = "GeoEnrichCore/1.0"

    def geocode_address(self, address: str, country: str = "Netherlands") -> Optional[Dict[str, Any]]:
        """
        Convert address to coordinates.

        Args:
            address: Street address or place name
            country: Country for context

        Returns:
            Dictionary with lat, lon, and full address
        """
        cache_key = f"{address}_{country}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Using Nominatim (OSM) - free, no API key required
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': f"{address}, {country}",
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': self.user_agent}

            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data:
                    result = {
                        'lat': float(data[0]['lat']),
                        'lon': float(data[0]['lon']),
                        'full_address': data[0].get('display_name', address),
                        'type': data[0].get('type', 'unknown')
                    }
                    self.cache[cache_key] = result
                    return result

            return None

        except Exception as e:
            print(f"⚠️ Geocoding error: {str(e)}")
            return None

    def reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
        """
        Convert coordinates to address.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Address string or None
        """
        cache_key = f"{lat:.6f}_{lon:.6f}"
        if cache_key in self.cache:
            return self.cache[cache_key].get('full_address')

        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1
            }
            headers = {'User-Agent': self.user_agent}

            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data and 'display_name' in data:
                    result = {
                        'full_address': data['display_name'],
                        'address': data.get('address', {})
                    }
                    self.cache[cache_key] = result
                    return result['full_address']

            return None

        except Exception as e:
            print(f"⚠️ Reverse geocoding error: {str(e)}")
            return None

    def geocode_batch(self, addresses: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Geocode multiple addresses.

        Args:
            addresses: List of address strings

        Returns:
            List of geocoding results
        """
        results = []
        for address in addresses:
            results.append(self.geocode_address(address))
        return results

    def autocomplete(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get address autocomplete suggestions.

        Args:
            query: Partial address or place name
            limit: Maximum number of suggestions

        Returns:
            List of suggestion dictionaries
        """
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': query,
                'format': 'json',
                'limit': limit,
                'addressdetails': 1
            }
            headers = {'User-Agent': self.user_agent}

            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data:
                    results.append({
                        'display_name': item.get('display_name', ''),
                        'lat': float(item.get('lat', 0)),
                        'lon': float(item.get('lon', 0)),
                        'type': item.get('type', 'unknown')
                    })
                return results

            return []

        except Exception as e:
            print(f"⚠️ Autocomplete error: {str(e)}")
            return []
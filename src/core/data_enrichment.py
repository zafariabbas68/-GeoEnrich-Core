"""
Data Enrichment Module
Enriches asset data with demographic and contextual information.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import requests


class DataEnrichment:
    """
    Enriches geospatial data with demographic and contextual information.
    """

    def __init__(self):
        self.population_data = None
        self.demographic_data = None
        self._load_default_data()

    def _load_default_data(self):
        """Load default demographic data."""
        # In production, this would load from real data sources
        # For demo, create sample demographic profiles
        self.demographic_data = {
            'urban_center': {
                'population_density': 4500,
                'avg_age': 35.2,
                'income_level': 'high',
                'employment_rate': 0.92,
                'education_level': 'university'
            },
            'residential': {
                'population_density': 2800,
                'avg_age': 42.7,
                'income_level': 'medium',
                'employment_rate': 0.78,
                'education_level': 'secondary'
            },
            'commercial': {
                'population_density': 1200,
                'avg_age': 38.5,
                'income_level': 'high',
                'employment_rate': 0.95,
                'education_level': 'university'
            },
            'industrial': {
                'population_density': 600,
                'avg_age': 44.1,
                'income_level': 'medium_low',
                'employment_rate': 0.82,
                'education_level': 'vocational'
            },
            'park': {
                'population_density': 300,
                'avg_age': 32.8,
                'income_level': 'medium',
                'employment_rate': 0.70,
                'education_level': 'secondary'
            }
        }

    def enrich_assets_with_demographics(self, assets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich assets with demographic data based on location.

        Args:
            assets_df: DataFrame with asset data

        Returns:
            Enriched DataFrame with demographic columns
        """
        if len(assets_df) == 0:
            return assets_df

        enriched = assets_df.copy()

        # Determine area type based on location (simplified)
        enriched['area_type'] = enriched.apply(
            lambda row: self._determine_area_type(row['lat'], row['lon']),
            axis=1
        )

        # Add demographic columns
        for area_type, demographics in self.demographic_data.items():
            mask = enriched['area_type'] == area_type
            for key, value in demographics.items():
                enriched.loc[mask, f'demo_{key}'] = value

        return enriched

    def _determine_area_type(self, lat: float, lon: float) -> str:
        """Determine area type based on coordinates."""
        # Simple heuristic based on location
        # In production, this would use actual land use data
        import random
        types = ['urban_center', 'residential', 'commercial', 'industrial', 'park']
        return random.choice(types)

    def generate_demographic_report(self, lat: float, lon: float, radius_km: float = 5) -> Dict[str, Any]:
        """
        Generate demographic report for an area.

        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Radius in kilometers

        Returns:
            Dictionary with demographic information
        """
        area_type = self._determine_area_type(lat, lon)
        demographics = self.demographic_data.get(area_type, {})

        return {
            'location': {'lat': lat, 'lon': lon},
            'radius_km': radius_km,
            'area_type': area_type,
            'demographics': demographics,
            'estimated_population': self._estimate_population(lat, lon, radius_km)
        }

    def _estimate_population(self, lat: float, lon: float, radius_km: float) -> int:
        """Estimate population in an area."""
        # Simple estimation for demo
        area_km2 = 3.14159 * (radius_km ** 2)
        density = 3000  # Average density per km2
        return int(area_km2 * density)

    def get_demographics_for_location(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get demographic data for a specific location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dictionary with demographic data
        """
        area_type = self._determine_area_type(lat, lon)
        return self.demographic_data.get(area_type, {})
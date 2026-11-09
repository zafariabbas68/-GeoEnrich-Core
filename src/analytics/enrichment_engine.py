

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime


class EnrichmentEngine:
    """
    Enriches mobile mapping data with additional intelligence.
    """

    @staticmethod
    def enrich_assets(assets_df: pd.DataFrame,
                      point_cloud: np.ndarray,
                      recognition_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich assets with recognition results and computed features.

        Args:
            assets_df: Original asset DataFrame
            point_cloud: Point cloud data
            recognition_df: Recognition results

        Returns:
            Enriched DataFrame
        """
        # Start with assets
        enriched = assets_df.copy()

        # Add recognition results
        if recognition_df is not None and len(recognition_df) > 0:
            # Merge on asset id
            recognition_subset = recognition_df[['id', 'recognized_type', 'confidence', 'is_valid']]
            enriched = enriched.merge(recognition_subset, left_on='id', right_on='id', how='left')

            # Rename columns
            enriched = enriched.rename(columns={
                'recognized_type': 'ai_type',
                'confidence': 'ai_confidence'
            })

        # Add computed attributes
        enriched['elevation_category'] = EnrichmentEngine._categorize_elevation(enriched['height_m'])
        enriched['size_category'] = EnrichmentEngine._categorize_size(enriched['height_m'])
        enriched['enrichment_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        enriched['data_quality_score'] = EnrichmentEngine._calculate_quality_score(enriched)

        return enriched

    @staticmethod
    def _categorize_elevation(height: float) -> str:
        """Categorize asset by elevation."""
        if height < 0.5:
            return 'Ground Level'
        elif height < 2.0:
            return 'Low (<2m)'
        elif height < 5.0:
            return 'Medium (2-5m)'
        elif height < 10.0:
            return 'High (5-10m)'
        else:
            return 'Very High (>10m)'

    @staticmethod
    def _categorize_size(height: float) -> str:
        """Categorize asset by size."""
        if height < 1.0:
            return 'Small'
        elif height < 3.0:
            return 'Medium'
        elif height < 6.0:
            return 'Large'
        else:
            return 'Extra Large'

    @staticmethod
    def _calculate_quality_score(enriched: pd.DataFrame) -> float:
        """Calculate data quality score for each asset."""
        # Placeholder - would use multiple factors in reality
        base_score = 0.8

        # Adjust based on confidence
        if 'ai_confidence' in enriched.columns:
            return round(enriched['ai_confidence'].fillna(0.5).mean(), 3)

        return base_score

    @staticmethod
    def generate_summary_statistics(enriched_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for enriched data.

        Args:
            enriched_df: Enriched DataFrame

        Returns:
            Dictionary with summary statistics
        """
        if len(enriched_df) == 0:
            return {}

        # Asset type distribution
        type_counts = enriched_df['type'].value_counts().to_dict()

        # Height statistics
        height_stats = {
            'min': enriched_df['height_m'].min(),
            'max': enriched_df['height_m'].max(),
            'mean': enriched_df['height_m'].mean(),
            'median': enriched_df['height_m'].median(),
            'std': enriched_df['height_m'].std()
        }

        # AI recognition statistics
        ai_stats = {}
        if 'ai_type' in enriched_df.columns:
            ai_counts = enriched_df['ai_type'].value_counts().to_dict()
            ai_confidence_mean = enriched_df['ai_confidence'].mean()
            valid_assets = len(enriched_df[enriched_df['is_valid'] == True])

            ai_stats = {
                'ai_type_counts': ai_counts,
                'avg_confidence': round(ai_confidence_mean, 3),
                'valid_assets': valid_assets,
                'invalid_assets': len(enriched_df) - valid_assets
            }

        return {
            'total_assets': len(enriched_df),
            'type_distribution': type_counts,
            'height_statistics': height_stats,
            'ai_statistics': ai_stats,
            'unique_types': len(type_counts)
        }

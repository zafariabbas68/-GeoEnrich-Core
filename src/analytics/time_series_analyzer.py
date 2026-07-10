"""
Time Series Analysis Module
Track asset changes over time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import random


class TimeSeriesAnalyzer:
    """
    Analyze asset changes over time.
    """

    def __init__(self):
        self.history = []
        self.timeline = []

    def generate_timeline(self, assets_df: pd.DataFrame,
                          n_steps: int = 10,
                          interval_days: int = 30) -> pd.DataFrame:
        """
        Generate time series data for assets.

        Args:
            assets_df: Current asset data
            n_steps: Number of time steps
            interval_days: Days between steps

        Returns:
            DataFrame with time series data
        """
        start_date = datetime.now() - timedelta(days=n_steps * interval_days)

        timeline_data = []

        for step in range(n_steps + 1):
            current_date = start_date + timedelta(days=step * interval_days)

            # Simulate changes over time
            assets_snapshot = self._simulate_snapshot(assets_df, step, n_steps)
            assets_snapshot['timestamp'] = current_date
            assets_snapshot['step'] = step

            timeline_data.append(assets_snapshot)

            # Store in history
            self.history.append(assets_snapshot)
            self.timeline.append(current_date)

        return pd.concat(timeline_data, ignore_index=True)

    def _simulate_snapshot(self, assets_df: pd.DataFrame,
                           step: int, total_steps: int) -> pd.DataFrame:
        """
        Simulate asset changes at a given time step.

        Args:
            assets_df: Base assets
            step: Current step
            total_steps: Total steps

        Returns:
            Modified assets at this time step
        """
        # Copy base data
        snapshot = assets_df.copy()

        # Add time-based changes
        progress = step / total_steps

        # Simulate changes
        changes = []

        for _, asset in assets_df.iterrows():
            # Random changes over time
            height_change = np.random.normal(0, 0.1 * progress)
            confidence_change = np.random.normal(0, 0.02 * progress)

            # Some assets appear/disappear
            if random.random() < 0.05 * progress:
                # Asset disappears
                continue

            # Modify asset
            modified = asset.copy()
            modified['height_m'] = max(0, modified['height_m'] + height_change)
            modified['confidence'] = min(0.99, max(0.7, modified['confidence'] + confidence_change))
            modified['step'] = step

            # Add status
            if step == 0:
                modified['status'] = 'existing'
            elif random.random() < 0.02 * progress:
                modified['status'] = 'new'
            elif random.random() < 0.03 * progress:
                modified['status'] = 'modified'
            else:
                modified['status'] = 'existing'

            changes.append(modified)

        return pd.DataFrame(changes)

    def detect_changes(self, snapshot1: pd.DataFrame,
                       snapshot2: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect changes between two snapshots.

        Args:
            snapshot1: First time snapshot
            snapshot2: Second time snapshot

        Returns:
            Dictionary with change detection results
        """
        # Match assets by ID
        ids1 = set(snapshot1['id'])
        ids2 = set(snapshot2['id'])

        # Find changes
        added = ids2 - ids1
        removed = ids1 - ids2
        common = ids1 & ids2

        # Analyze common assets
        modified = []
        unchanged = []

        for asset_id in common:
            asset1 = snapshot1[snapshot1['id'] == asset_id].iloc[0]
            asset2 = snapshot2[snapshot2['id'] == asset_id].iloc[0]

            if asset1['height_m'] != asset2['height_m']:
                modified.append({
                    'id': asset_id,
                    'type': asset1['type'],
                    'height_old': asset1['height_m'],
                    'height_new': asset2['height_m'],
                    'change': asset2['height_m'] - asset1['height_m']
                })
            else:
                unchanged.append(asset_id)

        # Calculate statistics
        added_assets = snapshot2[snapshot2['id'].isin(added)]
        removed_assets = snapshot1[snapshot1['id'].isin(removed)]

        return {
            'added': len(added),
            'removed': len(removed),
            'modified': len(modified),
            'unchanged': len(unchanged),
            'added_assets': added_assets.to_dict('records'),
            'removed_assets': removed_assets.to_dict('records'),
            'modified_assets': modified,
            'total_change': len(added) + len(removed) + len(modified)
        }

    def analyze_trends(self, timeline_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trends over time.

        Args:
            timeline_df: Time series data

        Returns:
            Trend analysis results
        """
        if len(timeline_df) == 0:
            return {}

        # Group by step and asset type
        trends = timeline_df.groupby(['step', 'type']).agg({
            'height_m': ['mean', 'std', 'count'],
            'confidence': 'mean'
        }).reset_index()

        # Calculate growth rates
        steps = sorted(timeline_df['step'].unique())
        growth_rates = {}

        for asset_type in timeline_df['type'].unique():
            type_data = timeline_df[timeline_df['type'] == asset_type]

            if len(type_data) == 0:
                continue

            # Calculate average height over time
            avg_heights = type_data.groupby('step')['height_m'].mean()

            if len(avg_heights) > 1:
                # Calculate growth rate
                start = avg_heights.iloc[0]
                end = avg_heights.iloc[-1]
                growth = ((end - start) / start) * 100 if start > 0 else 0

                growth_rates[asset_type] = {
                    'growth_percent': growth,
                    'start_height': start,
                    'end_height': end,
                    'steps': len(avg_heights)
                }

        return {
            'trends': trends,
            'growth_rates': growth_rates,
            'total_steps': len(steps),
            'asset_types': timeline_df['type'].unique().tolist()
        }

    def generate_forecast(self, timeline_df: pd.DataFrame,
                          steps_ahead: int = 5) -> pd.DataFrame:
        """
        Generate simple forecast using linear regression.

        Args:
            timeline_df: Time series data
            steps_ahead: Number of steps to forecast

        Returns:
            Forecast DataFrame
        """
        if len(timeline_df) == 0:
            return pd.DataFrame()

        forecasts = []

        for asset_type in timeline_df['type'].unique():
            type_data = timeline_df[timeline_df['type'] == asset_type]

            if len(type_data) < 3:
                continue

            # Simple linear regression
            steps = type_data['step'].values
            heights = type_data['height_m'].values

            if len(steps) > 1:
                # Calculate trend
                slope, intercept = np.polyfit(steps, heights, 1)

                # Forecast
                last_step = max(steps)
                for i in range(1, steps_ahead + 1):
                    forecast_step = last_step + i
                    forecast_height = slope * forecast_step + intercept

                    forecasts.append({
                        'type': asset_type,
                        'step': forecast_step,
                        'forecast_height_m': max(0, forecast_height),
                        'confidence': 0.85 - (i * 0.05),
                        'is_forecast': True
                    })

        return pd.DataFrame(forecasts)

    def get_change_summary(self, snapshot1: pd.DataFrame,
                           snapshot2: pd.DataFrame) -> str:
        """
        Generate a human-readable change summary.

        Args:
            snapshot1: First time snapshot
            snapshot2: Second time snapshot

        Returns:
            Summary string
        """
        changes = self.detect_changes(snapshot1, snapshot2)

        summary = f"""
        📊 Change Summary
        =================
        Total Assets: {len(snapshot1)} → {len(snapshot2)}
        ✅ Added: {changes['added']}
        ❌ Removed: {changes['removed']}
        ✏️ Modified: {changes['modified']}
        ⚪ Unchanged: {changes['unchanged']}
        Total Changes: {changes['total_change']}
        """

        if changes['modified_assets']:
            summary += "\nModified Assets:\n"
            for asset in changes['modified_assets'][:5]:
                summary += f"  - {asset['type']} {asset['id']}: {asset['height_old']:.2f}m → {asset['height_new']:.2f}m ({asset['change']:+.2f}m)\n"

        return summary
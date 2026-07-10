#!/usr/bin/env python3
"""
Full pipeline demo for GeoEnrich Core.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

from src.core.data_generation import MobileMappingGenerator
from src.analytics.spatial_analytics import SpatialAnalytics
from src.analytics.asset_recognizer import AssetRecognizer


def run_pipeline():
    print("🚀 GeoEnrich Core - Full Pipeline Demo")
    print("=" * 50)

    # Step 1: Generate data
    print("\n📊 Step 1: Generating mobile mapping data...")
    generator = MobileMappingGenerator()
    files = generator.generate_full_dataset('outputs/sample_data')
    print(f"   ✅ Generated {len(generator.assets)} assets")
    print(f"   ✅ Generated {len(generator.point_cloud)} LiDAR points")

    # Load data
    assets_df = pd.read_csv(files['assets_path'])
    point_cloud = np.load(files['point_cloud_path'], allow_pickle=True)

    # Step 2: Spatial analysis
    print("\n🔬 Step 2: Running spatial analysis...")
    proximity = SpatialAnalytics.calculate_proximity(assets_df)
    clusters = SpatialAnalytics.cluster_assets(assets_df)
    distribution = SpatialAnalytics.analyze_distribution(assets_df)

    print(f"   ✅ Found {len(clusters['cluster_id'].unique())} asset clusters")
    print(f"   ✅ Mean distance: {distribution['mean_distance_km']:.4f} km")
    print(f"   ✅ Isolated assets: {len(proximity[proximity['is_isolated']])}")

    # Step 3: Asset recognition
    print("\n🤖 Step 3: Running AI asset recognition...")
    recognizer = AssetRecognizer()
    recognized = recognizer.recognize_assets(assets_df, point_cloud)
    recognition_df = recognizer.generate_report()

    valid = len(recognition_df[recognition_df['is_valid']])
    print(f"   ✅ Recognized {valid}/{len(recognition_df)} assets ({valid / len(recognition_df) * 100:.1f}%)")

    # Step 4: Summary
    print("\n📊 Summary Statistics:")
    print(f"   - Total Assets: {len(assets_df)}")
    print(f"   - Asset Types: {len(assets_df['type'].unique())}")
    print(f"   - Clusters: {len(clusters['cluster_id'].unique())}")
    print(f"   - Mean Height: {assets_df['height_m'].mean():.2f} m")
    print(f"   - Max Height: {assets_df['height_m'].max():.2f} m")

    print("\n✨ Pipeline complete!")
    print(f"\nRun: streamlit run src/visualization/dashboard.py")
    print("To view the interactive dashboard.")


if __name__ == "__main__":
    run_pipeline()
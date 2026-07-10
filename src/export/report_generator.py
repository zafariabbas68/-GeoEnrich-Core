"""
Report Generation Module
Creates professional reports from enriched data.
"""

import os
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any


def generate_html_report(enriched_df: pd.DataFrame,
                         summary: Dict[str, Any],
                         output_path: str = "outputs/reports/report.html") -> str:
    """
    Generate HTML report from enriched data.

    Args:
        enriched_df: Enriched DataFrame
        summary: Summary statistics
        output_path: Output file path

    Returns:
        Path to created report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Asset type distribution
    type_counts = enriched_df['type'].value_counts()

    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>GeoEnrich Core - Asset Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #1a365d; }}
            .summary {{ background: #f0f4ff; padding: 20px; border-radius: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #667eea; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>🌍 GeoEnrich Core - Asset Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <h2>Executive Summary</h2>
            <p><strong>Total Assets:</strong> {summary['total_assets']}</p>
            <p><strong>Asset Types:</strong> {summary['unique_types']}</p>
            <p><strong>Mean Height:</strong> {summary['height_statistics']['mean']:.2f} m</p>
            <p><strong>Max Height:</strong> {summary['height_statistics']['max']:.2f} m</p>
        </div>

        <h2>Asset Type Distribution</h2>
        <table>
            <tr><th>Type</th><th>Count</th><th>Percentage</th></tr>
    """

    total = len(enriched_df)
    for asset_type, count in type_counts.items():
        pct = (count / total) * 100
        html += f"<tr><td>{asset_type}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"

    html += """
        </table>

        <h2>All Assets</h2>
        <table>
            <tr><th>ID</th><th>Type</th><th>Height (m)</th><th>AI Type</th><th>Confidence</th></tr>
    """

    for _, row in enriched_df.iterrows():
        ai_type = row.get('ai_type', '-')
        confidence = row.get('ai_confidence', '-')
        html += f"""
            <tr>
                <td>{row['id']}</td>
                <td>{row['type']}</td>
                <td>{row['height_m']:.2f}</td>
                <td>{ai_type}</td>
                <td>{confidence if confidence == '-' else f'{confidence:.2f}'}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def generate_json_report(enriched_df: pd.DataFrame,
                         summary: Dict[str, Any],
                         output_path: str = "outputs/reports/report.json") -> str:
    """
    Generate JSON report from enriched data.

    Args:
        enriched_df: Enriched DataFrame
        summary: Summary statistics
        output_path: Output file path

    Returns:
        Path to created report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert DataFrame to records
    records = enriched_df.to_dict(orient='records')

    report = {
        'generated': datetime.now().isoformat(),
        'summary': summary,
        'assets': records
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return output_path
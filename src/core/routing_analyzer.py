"""
Routing Analysis Module
Provides routing, directions, and network analysis capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from shapely.geometry import Point, LineString
import math

try:
    import networkx as nx
    import osmnx as ox

    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("⚠️ OSMnx/NetworkX not installed. Install with: pip install osmnx networkx")


class RoutingAnalyzer:
    """
    Routing and network analysis for geospatial data.
    """

    def __init__(self):
        self.graph = None
        self.roads = None
        self._graph_built = False

    def build_road_network(self, bbox: Dict[str, float]) -> bool:
        """
        Build road network from OSM.

        Args:
            bbox: {'min_lat', 'max_lat', 'min_lon', 'max_lon'}

        Returns:
            True if successful
        """
        if not OSMNX_AVAILABLE:
            print("⚠️ OSMnx not available. Using simplified network.")
            self._build_simplified_network()
            return True

        try:
            self.graph = ox.graph_from_bbox(
                bbox['min_lat'], bbox['max_lat'],
                bbox['min_lon'], bbox['max_lon'],
                network_type='drive'
            )
            self._graph_built = True
            print(f"✅ Built road network with {len(self.graph.nodes)} nodes")
            return True
        except Exception as e:
            print(f"⚠️ Failed to build road network: {str(e)}")
            self._build_simplified_network()
            return True

    def _build_simplified_network(self):
        """Build a simplified road network for demo purposes."""
        self.graph = nx.Graph()

        # Create a grid of nodes
        center_lat, center_lon = 52.3676, 4.9041
        for i in range(-5, 6):
            for j in range(-5, 6):
                lat = center_lat + i * 0.002
                lon = center_lon + j * 0.002
                node_id = f"{i}_{j}"
                self.graph.add_node(node_id, lat=lat, lon=lon)

                # Add edges to neighbors
                if i > -5:
                    neighbor = f"{i - 1}_{j}"
                    self.graph.add_edge(node_id, neighbor, weight=0.15)
                if j > -5:
                    neighbor = f"{i}_{j - 1}"
                    self.graph.add_edge(node_id, neighbor, weight=0.15)

        self._graph_built = True

    def find_shortest_route(self, origin: Tuple[float, float],
                            destination: Tuple[float, float]) -> Dict[str, Any]:
        """
        Find shortest route between two points.

        Args:
            origin: (lat, lon) tuple
            destination: (lat, lon) tuple

        Returns:
            Dictionary with route information
        """
        if not self._graph_built:
            self._build_simplified_network()

        # Find nearest nodes
        origin_node = self._find_nearest_node(origin)
        dest_node = self._find_nearest_node(destination)

        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, origin_node, dest_node, weight='weight')

            # Get coordinates along path
            route_points = []
            for node in path:
                route_points.append({
                    'lat': self.graph.nodes[node]['lat'],
                    'lon': self.graph.nodes[node]['lon']
                })

            # Calculate route statistics
            total_distance = self._calculate_path_length(path)

            return {
                'route_points': route_points,
                'total_distance_km': round(total_distance, 3),
                'estimated_time_min': round(total_distance * 2, 1),  # 30 km/h average
                'node_count': len(path),
                'start_node': origin_node,
                'end_node': dest_node
            }
        except nx.NetworkXNoPath:
            return {
                'route_points': [],
                'total_distance_km': -1,
                'estimated_time_min': -1,
                'error': 'No path found'
            }

    def _find_nearest_node(self, point: Tuple[float, float]) -> str:
        """Find nearest graph node to a point."""
        lat, lon = point

        # For simplified network, find closest node by coordinates
        min_dist = float('inf')
        nearest = None

        for node in self.graph.nodes:
            node_lat = self.graph.nodes[node].get('lat', 0)
            node_lon = self.graph.nodes[node].get('lon', 0)
            dist = ((lat - node_lat) ** 2 + (lon - node_lon) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def _calculate_path_length(self, path: List[str]) -> float:
        """Calculate total path length in km."""
        total = 0
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            lat1 = self.graph.nodes[node1]['lat']
            lon1 = self.graph.nodes[node1]['lon']
            lat2 = self.graph.nodes[node2]['lat']
            lon2 = self.graph.nodes[node2]['lon']
            total += self._haversine(lat1, lon1, lat2, lon2)
        return total

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km."""
        R = 6371
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) *
             math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def generate_service_area(self, center: Tuple[float, float],
                              radius_km: float) -> Dict[str, Any]:
        """
        Generate a service area around a location.

        Args:
            center: (lat, lon) tuple
            radius_km: Radius in kilometers

        Returns:
            Dictionary with service area information
        """
        lat, lon = center

        # Generate points on a circle
        points = []
        num_points = 36
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dlat = (radius_km / 111.32) * math.cos(angle)
            dlon = (radius_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(angle)
            points.append({
                'lat': lat + dlat,
                'lon': lon + dlon
            })

        # Add center point
        points.append({'lat': lat, 'lon': lon})

        return {
            'center': {'lat': lat, 'lon': lon},
            'radius_km': radius_km,
            'boundary_points': points,
            'area_km2': math.pi * (radius_km ** 2)
        }

    def optimize_multistop_route(self, stops: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Optimize route with multiple stops.

        Args:
            stops: List of (lat, lon) tuples

        Returns:
            Dictionary with optimized route
        """
        if len(stops) < 3:
            return self.find_shortest_route(stops[0], stops[-1])

        # Simple optimization: nearest neighbor
        unvisited = list(range(1, len(stops)))
        current = 0
        route = [0]

        while unvisited:
            nearest = min(unvisited, key=lambda i: self._haversine(
                stops[current][0], stops[current][1],
                stops[i][0], stops[i][1]
            ))
            unvisited.remove(nearest)
            route.append(nearest)
            current = nearest

        # Build route points
        route_points = []
        for idx in route:
            route_points.append({
                'lat': stops[idx][0],
                'lon': stops[idx][1]
            })

        return {
            'route_points': route_points,
            'stop_order': route,
            'total_distance_km': round(sum(
                self._haversine(stops[route[i]][0], stops[route[i]][1],
                                stops[route[i + 1]][0], stops[route[i + 1]][1])
                for i in range(len(route) - 1)
            ), 3)
        }
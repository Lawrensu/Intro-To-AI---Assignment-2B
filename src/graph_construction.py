"""
Graph construction from road network for GCN
Builds spatial graph representation from road network data
Author: Cherylynn
"""

import torch
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def parse_road_network(file_path: str) -> Tuple[Dict, List, Dict, Dict]:
    """
    Parse road network file
    
    Args:
        file_path: Path to network file
    
    Returns:
        tuple: (nodes, ways, cameras, meta)
    """
    nodes = {}
    ways = []
    cameras = {}
    meta = {}
    
    section = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('[') and line.endswith(']'):
                section = line.upper()
                continue
            
            parts = [p.strip() for p in line.split(',')]
            
            if section == '[NODES]':
                node_id, lat, lon, label = parts[0], float(parts[1]), float(parts[2]), parts[3]
                nodes[node_id] = {
                    'lat': lat,
                    'lon': lon,
                    'label': label
                }
            
            elif section == '[WAYS]':
                way = {
                    'way_id': parts[0],
                    'from': parts[1],
                    'to': parts[2],
                    'road_name': parts[3],
                    'highway_type': parts[4],
                    'time_min': float(parts[5])
                }
                ways.append(way)
            
            elif section == '[CAMERAS]':
                cameras[parts[0]] = parts[1]
            
            elif section == '[META]':
                key = parts[0].upper()
                if key == 'START':
                    meta['start'] = parts[1]
                elif key == 'GOAL':
                    meta['goals'] = parts[1:]
                elif key == 'ACCIDENT_MULTIPLIER':
                    meta['accident_multiplier'] = float(parts[1])
    
    return nodes, ways, cameras, meta


def construct_graph(nodes: Dict, ways: List) -> Data:
    """
    Construct PyG Data object from road network
    
    Args:
        nodes: Dictionary of nodes
        ways: List of road segments (edges)
    
    Returns:
        Data: PyG Data object with node features, edge_index, labels, and masks
    """
    # Create node ID mapping
    node_list = sorted(nodes.keys())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    num_nodes = len(node_list)
    
    print(f"\nConstructing graph...")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {len(ways)}")
    
    # Node features: [lat, lon, normalized_lat, normalized_lon, degree]
    lats = np.array([nodes[nid]['lat'] for nid in node_list])
    lons = np.array([nodes[nid]['lon'] for nid in node_list])
    
    # Normalize coordinates
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    norm_lats = (lats - lat_min) / (lat_max - lat_min + 1e-8)
    norm_lons = (lons - lon_min) / (lon_max - lon_min + 1e-8)
    
    # Initialize degree counter
    degrees = np.zeros(num_nodes)
    
    # Construct edge list
    edge_list = []
    edge_attr_list = []  # Travel time
    
    for way in ways:
        from_node = way['from']
        to_node = way['to']
        
        if from_node in node_to_idx and to_node in node_to_idx:
            from_idx = node_to_idx[from_node]
            to_idx = node_to_idx[to_node]
            
            # Add edge
            edge_list.append([from_idx, to_idx])
            edge_attr_list.append(way['time_min'])
            
            # Update degrees
            degrees[from_idx] += 1
            degrees[to_idx] += 1
    
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1)
    
    # Calculate additional features
    # Betweenness centrality (how many shortest paths pass through this node)
    betweenness = np.zeros(num_nodes)
    for i in range(num_nodes):
        # Simple approximation: nodes with higher degree are more central
        betweenness[i] = degrees[i] / (num_nodes - 1) if num_nodes > 1 else 0
    
    # Closeness centrality (inverse of average distance to all other nodes)
    closeness = np.zeros(num_nodes)
    for i in range(num_nodes):
        # Approximation based on degree
        closeness[i] = degrees[i] / max(degrees) if max(degrees) > 0 else 0
    
    # Distance from center (geographic)
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    dist_from_center = np.sqrt((lats - center_lat)**2 + (lons - center_lon)**2)
    norm_dist_from_center = (dist_from_center - dist_from_center.min()) / (dist_from_center.max() - dist_from_center.min() + 1e-8)
    
    # Degree normalized
    norm_degree = degrees / (max(degrees) + 1e-8)
    
    # Create node features: [lat, lon, norm_lat, norm_lon, degree, norm_degree, betweenness, closeness, dist_from_center]
    node_features = np.stack([
        lats,
        lons,
        norm_lats,
        norm_lons,
        degrees,
        norm_degree,
        betweenness,
        closeness,
        norm_dist_from_center
    ], axis=1)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create more realistic labels based on node importance
    # Combine degree, centrality, and location to determine traffic level
    # Tourist/cultural sites + high connectivity = high traffic
    # NOTE: With only 15 nodes, we use a hybrid approach based on:
    # - Connectivity (degree)
    # - Location type (tourist attractions vs others)
    # - Geographic centrality
    y = torch.zeros(num_nodes, dtype=torch.long)
    
    # Define important heritage/tourist nodes (from node labels)
    # These are major attractions that typically have high traffic
    high_traffic_keywords = ['Waterfront', 'Plaza', 'Museum', 'Courthouse', 'Cathedral', 'Padang Merdeka']
    medium_traffic_keywords = ['Masjid', 'Art', 'Junction', 'Car Park', 'Wisma', 'Monument']
    
    for i, node_id in enumerate(node_list):
        node_label = nodes[node_id]['label']
        node_degree = degrees[i]
        is_central = norm_dist_from_center[i] < 0.3  # Central location
        
        # Determine traffic level based on location type and connectivity
        is_high_traffic = any(keyword in node_label for keyword in high_traffic_keywords)
        is_medium_traffic = any(keyword in node_label for keyword in medium_traffic_keywords)
        
        # More nuanced classification
        if (is_high_traffic and node_degree >= 4) or (node_degree >= 6):
            y[i] = 2  # High traffic (major attractions/hubs)
        elif is_high_traffic or is_medium_traffic or node_degree >= 3:
            y[i] = 1  # Medium traffic (moderate importance)
        else:
            y[i] = 0  # Low traffic (minor locations)
    
    # Create stratified train/val/test masks to ensure balanced class distribution
    from sklearn.model_selection import train_test_split
    
    # Get indices for each class
    indices_array = np.arange(num_nodes)
    y_numpy = y.numpy()
    
    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices_array, 
        test_size=0.3, 
        stratify=y_numpy,
        random_state=42
    )
    
    # Second split: val vs test
    y_temp = y_numpy[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # Store node mapping for later use
    data.node_list = node_list
    data.node_to_idx = node_to_idx
    
    print(f"✓ Graph constructed successfully")
    print(f"  Node features shape: {x.shape} (9 features per node)")
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    print(f"  Class distribution: Low={sum(y==0)}, Medium={sum(y==1)}, High={sum(y==2)}")
    print(f"  Train classes: Low={sum(y[train_mask]==0)}, Med={sum(y[train_mask]==1)}, High={sum(y[train_mask]==2)}")
    print(f"  Val classes: Low={sum(y[val_mask]==0)}, Med={sum(y[val_mask]==1)}, High={sum(y[val_mask]==2)}")
    print(f"  Test classes: Low={sum(y[test_mask]==0)}, Med={sum(y[test_mask]==1)}, High={sum(y[test_mask]==2)}")
    
    return data


if __name__ == "__main__":
    print("Testing graph construction...")
    
    # Test with actual network file
    network_file = "heritage_assignment_15_time_asymmetric-1.txt"
    
    if Path(network_file).exists():
        nodes, ways, cameras, meta = parse_road_network(network_file)
        print(f"\nParsed network:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Ways: {len(ways)}")
        print(f"  Cameras: {len(cameras)}")
        print(f"  Meta: {meta}")
        
        data = construct_graph(nodes, ways)
        
        print(f"\n✓ Graph construction test successful!")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Number of edges: {data.num_edges}")
        print(f"  Number of features: {data.num_node_features}")
    else:
        print(f"Network file not found: {network_file}")

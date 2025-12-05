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
    
    # Create node features: [lat, lon, norm_lat, norm_lon, degree]
    node_features = np.stack([
        lats,
        lons,
        norm_lats,
        norm_lons,
        degrees
    ], axis=1)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create labels based on node degree (traffic flow proxy)
    # Low traffic (0): degree <= 2
    # Medium traffic (1): degree 3-4
    # High traffic (2): degree >= 5
    y = torch.zeros(num_nodes, dtype=torch.long)
    for i, degree in enumerate(degrees):
        if degree <= 2:
            y[i] = 0  # Low traffic (Minor)
        elif degree <= 4:
            y[i] = 1  # Medium traffic (Moderate)
        else:
            y[i] = 2  # High traffic (Severe)
    
    # Create train/val/test masks (70/15/15 split)
    indices = torch.randperm(num_nodes)
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
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
    print(f"  Node features shape: {x.shape}")
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    print(f"  Class distribution: Low={sum(y==0)}, Medium={sum(y==1)}, High={sum(y==2)}")
    
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

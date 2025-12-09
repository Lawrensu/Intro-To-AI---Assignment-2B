"""
Integration module: Combines ML predictions with pathfinding from Part A
"""

import torch
from pathlib import Path
from typing import List, Tuple, Dict
from src.models.cnn_model import load_cnn_model
import heapq


# Severity multipliers for travel time adjustment
SEVERITY_MULTIPLIERS = {
    'none': 1.0,      # No adjustment
    'minor': 1.2,     # 20% increase
    'moderate': 1.5,  # 50% increase
    'severe': 2.0     # 100% increase
}


class IntegratedICS:
    """
    Integrated Traffic Incident Classification System
    Combines ML predictions with pathfinding
    """
    
    def __init__(self, cnn_model_path: str, network_file: str):
        """
        Initialize the integrated system
        
        Args:
            cnn_model_path: Path to trained CNN model
            network_file: Path to road network file from Part A
        """
        # Load ML model
        self.cnn_model = load_cnn_model(cnn_model_path)
        self.cnn_model.eval()
        
        # Load road network
        self.graph = self.load_road_network(network_file)
        self.nodes = self.graph['nodes']
        self.edges = self.graph['edges']
        
        print("✓ Integrated ICS initialized")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Edges: {len(self.edges)}")
    
    def load_road_network(self, file_path: str) -> Dict:
        """Load road network from Part A"""
        # TODO: Import from your Part A implementation
        pass
    
    def predict_severity(self, image_path: str) -> Tuple[str, float]:
        """
        Predict incident severity from image
        
        Args:
            image_path: Path to incident image
        
        Returns:
            Tuple of (severity_class, confidence)
        """
        # TODO: Implement image preprocessing and prediction
        pass
    
    def adjust_travel_time(self, base_time: float, severity: str) -> float:
        """
        Adjust travel time based on predicted severity
        
        Args:
            base_time: Base travel time (minutes)
            severity: Predicted severity class
        
        Returns:
            Adjusted travel time
        """
        multiplier = SEVERITY_MULTIPLIERS.get(severity, 1.0)
        return base_time * multiplier
    
    def find_top_k_paths(self, origin: str, destination: str, k: int = 5) -> List[Dict]:
        """
        Find top-k shortest paths with incident-adjusted times
        
        Args:
            origin: Origin landmark
            destination: Destination landmark
            k: Number of paths to return
        
        Returns:
            List of path dictionaries with adjusted times
        """
        # TODO: Implement Yen's k-shortest paths or similar
        # Integrate with Part A pathfinding
        pass


if __name__ == "__main__":
    # Test integration
    ics = IntegratedICS(
        cnn_model_path="models/cnn_model.pth",
        network_file="data/heritage_assignment_15_time_asymmetric-1.txt"
    )
    
    # Test prediction
    severity, confidence = ics.predict_severity("test_image.jpg")
    print(f"Predicted: {severity} ({confidence:.2%})")
    
    # Test pathfinding
    paths = ics.find_top_k_paths("START", "GOAL", k=5)
    for i, path in enumerate(paths, 1):
        print(f"Path {i}: {path['time']:.2f} min")
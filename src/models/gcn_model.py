"""
Graph Convolutional Network (GCN) Model for Traffic Flow Prediction
Implements spatial relationship modeling in road networks using PyTorch Geometric
Author: Cherylynn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple, Optional

# Model configuration
HIDDEN_DIM = 16
NUM_CLASSES = 3  # Low, Medium, High traffic flow levels
DROPOUT = 0.3


class TrafficGCN(nn.Module):
    """
    2-layer Graph Convolutional Network for traffic flow prediction
    
    Args:
        num_node_features (int): Number of input features per node
        hidden_dim (int): Hidden layer dimension
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
    """
    
    def __init__(self, 
                 num_node_features: int,
                 hidden_dim: int = HIDDEN_DIM,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT):
        super(TrafficGCN, self).__init__()
        
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        # GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data object with x (node features) and edge_index
        
        Returns:
            torch.Tensor: Node-level predictions (log softmax)
        """
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
    def predict(self, data):
        """
        Make predictions
        
        Args:
            data: PyG Data object
        
        Returns:
            torch.Tensor: Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(data)
            pred = out.argmax(dim=1)
        return pred


class GCNTrainer:
    """
    Trainer class for GCN model
    Handles training loop, validation, and model checkpointing
    """
    
    def __init__(self, 
                 model: TrafficGCN,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.01):
        """
        Initialize trainer
        
        Args:
            model: TrafficGCN model
            device: Device to train on
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=5e-4
        )
        self.best_val_acc = 0.0
        self.best_model_wts = model.state_dict().copy()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, data, train_mask):
        """
        Train for one epoch
        
        Args:
            data: PyG Data object
            train_mask: Boolean mask for training nodes
        
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct = (pred[train_mask] == data.y[train_mask]).sum()
        acc = int(correct) / int(train_mask.sum())
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(self, data, mask):
        """
        Evaluate on validation or test set
        
        Args:
            data: PyG Data object
            mask: Boolean mask for evaluation nodes
        
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.eval()
        
        out = self.model(data)
        loss = F.nll_loss(out[mask], data.y[mask])
        
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        
        return loss.item(), acc
    
    def train(self,
              data,
              train_mask,
              val_mask,
              num_epochs: int = 200,
              patience: int = 20,
              save_path: str = 'models/gcn_model.pth'):
        """
        Train the GCN model
        
        Args:
            data: PyG Data object
            train_mask: Training node mask
            val_mask: Validation node mask
            num_epochs: Number of training epochs
            patience: Early stopping patience
            save_path: Path to save best model
        
        Returns:
            dict: Training history
        """
        print(f"\nTraining GCN model on {self.device}...")
        print(f"Training nodes: {train_mask.sum()}")
        print(f"Validation nodes: {val_mask.sum()}")
        print("=" * 70)
        
        data = data.to(self.device)
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(data, train_mask)
            
            # Validate
            val_loss, val_acc = self.evaluate(data, val_mask)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_wts = self.model.state_dict().copy()
                epochs_no_improve = 0
                print(f"  *** New best model! (Val Acc: {val_acc:.4f}) ***")
                
                # Save checkpoint
                self.save_model(save_path)
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(self.best_model_wts)
        
        return self.history
    
    def save_model(self, path: str):
        """
        Save model checkpoint
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']


def load_gcn_model(path: str, num_node_features: int, device: str = 'cuda'):
    """
    Load trained GCN model
    
    Args:
        path: Path to model checkpoint
        num_node_features: Number of input features
        device: Device to load model on
    
    Returns:
        TrafficGCN: Loaded model
    """
    device = device if torch.cuda.is_available() else 'cpu'
    model = TrafficGCN(num_node_features=num_node_features)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    print("Testing GCN model...")
    
    # Create dummy graph data
    num_nodes = 100
    num_features = 5
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    y = torch.randint(0, NUM_CLASSES, (num_nodes,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Create model
    model = TrafficGCN(num_node_features=num_features)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    
    print(f"✓ GCN model test successful!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Predictions sample: {pred[:10]}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")

"""
CNN Model for Traffic Incident Severity Classification

Uses ResNet-18 architecture with transfer learning for 3-class classification:
- Class 0: Minor
- Class 1: Moderate
- Class 2: Severe

Features:
- Pre-trained ResNet-18 backbone
- Fine-tuned final layer for 3-class classification
- Support for GPU acceleration
- Model checkpointing and loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import Dict, Optional
import time
import copy
from pathlib import Path

# Number of classes for incident severity
NUM_CLASSES = 4  
CLASS_NAMES = ['none', 'minor', 'moderate', 'severe'] 


class IncidentCNN(nn.Module):
    """
    CNN model for traffic incident classification using ResNet-18
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        """
        Initialize the CNN model
        
        Args:
            num_classes (int): Number of output classes (default: 3)
            pretrained (bool): Use pretrained ImageNet weights (default: True)
        """
        super(IncidentCNN, self).__init__()
        
        # Load pre-trained ResNet-18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Get the number of features in the final layer
        num_features = self.model.fc.in_features
        
        # Replace the final fully connected layer
        # ResNet-18 original: 512 -> 1000 (ImageNet classes)
        # Our model: 512 -> 3 (incident severity classes)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)


class CNNTrainer:
    """
    Trainer class for the CNN model
    """
    
    def __init__(self, model: IncidentCNN, device: str = 'cuda'):
        """
        Initialize the trainer
        
        Args:
            model: The CNN model to train
            device: Device to train on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.best_model_wts = None
        self.best_acc = 0.0
    
    def train_epoch(self, dataloader, criterion, optimizer) -> tuple:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, dataloader, criterion) -> tuple:
        """
        Validate for one epoch
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def train(self, 
              dataloaders: Dict,
              num_epochs: int = 25,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              class_weights: Optional[torch.Tensor] = None,
              patience: int = 5,
              save_path: str = 'models/cnn_model.pth'):
        """
        Train the model
        
        Args:
            dataloaders: Dictionary with 'train' and 'val' dataloaders
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            class_weights: Optional class weights for handling imbalance
            patience: Early stopping patience
            save_path: Path to save the best model
        
        Returns:
            dict: Training history
        """
        print(f"\nTraining CNN model on {self.device}...")
        print(f"Number of classes: {NUM_CLASSES}")
        print(f"Class names: {CLASS_NAMES}")
        print(f"Training samples: {len(dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(dataloaders['val'].dataset)}")
        
        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted loss with class weights: {class_weights.cpu().numpy()}")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), 
                              lr=learning_rate,
                              weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3, verbose=True
        )
        
        # Training loop
        since = time.time()
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train phase
            train_loss, train_acc = self.train_epoch(
                dataloaders['train'], criterion, optimizer
            )
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(
                dataloaders['val'], criterion
            )
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch results
            print(f'\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s)')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > self.best_acc:
                print(f'  *** New best model! (Val Acc: {val_acc:.4f}) ***')
                self.best_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                
                # Save model checkpoint
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                self.save_model(save_path)
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
        
        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed / 60:.2f} minutes')
        print(f'Best validation accuracy: {self.best_acc:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)
        
        return self.history
    
    def evaluate(self, dataloader, class_names: list = CLASS_NAMES) -> Dict:
        """
        Evaluate the model on a dataset
        
        Args:
            dataloader: Data loader for evaluation
            class_names: List of class names
        
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'per_class_metrics': {
                class_names[i]: {
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'f1_score': per_class_f1[i]
                }
                for i in range(len(class_names))
            },
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        return results
    
    def save_model(self, path: str):
        """
        Save model checkpoint
        
        Args:
            path: Path to save the model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES
        }, path)
        print(f'Model saved to {path}')
    
    def load_model(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.history = checkpoint.get('history', {})
        print(f'Model loaded from {path}')
        print(f'Best accuracy: {self.best_acc:.4f}')


def create_cnn_model(pretrained: bool = True, device: str = 'cuda') -> IncidentCNN:
    """
    Create a CNN model for traffic incident classification
    
    Args:
        pretrained: Use pretrained ImageNet weights
        device: Device to place the model on
    
    Returns:
        IncidentCNN model
    """
    model = IncidentCNN(num_classes=NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)
    
    print(f"Created ResNet-18 CNN model for {NUM_CLASSES}-class classification")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Device: {device}")
    
    return model


def load_cnn_model(path: str, device: str = 'cuda') -> IncidentCNN:
    """
    Load a trained CNN model from checkpoint
    
    Args:
        path: Path to the model checkpoint
        device: Device to place the model on
    
    Returns:
        Loaded IncidentCNN model
    """
    model = create_cnn_model(pretrained=False, device=device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded CNN model from {path}")
    print(f"Model accuracy: {checkpoint.get('best_acc', 'N/A')}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing CNN model...")
    
    # Create model
    model = create_cnn_model(pretrained=True, device='cpu')
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test predictions
    probs = torch.softmax(output, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    print(f"\nPredictions: {preds}")
    print(f"Predicted classes: {[CLASS_NAMES[p] for p in preds]}")
    print(f"\n✓ CNN model test successful!")
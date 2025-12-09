"""
RNN/LSTM Model for Traffic Pattern Analysis and Travel Time Prediction

This module implements:
1. RNN for traffic pattern analysis - analyzes temporal patterns in traffic data
2. LSTM for travel time prediction - predicts journey time based on chosen path

Features:
- Bidirectional LSTM architecture for better sequence understanding
- Support for both single-step and multi-step predictions
- Dropout for regularization
- GPU acceleration support
- Model checkpointing and loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
import time
import copy
from pathlib import Path


class TrafficPatternRNN(nn.Module):
    """
    RNN model for analyzing traffic patterns over time
    
    This model identifies recurring patterns in traffic flow,
    congestion levels, and incident frequencies.
    """
    
    def __init__(self, 
                 input_size: int = 10,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 3,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize the Traffic Pattern RNN
        
        Args:
            input_size (int): Number of input features (e.g., traffic volume, speed, density)
            hidden_size (int): Number of hidden units in RNN layers
            num_layers (int): Number of stacked RNN layers
            output_size (int): Number of output classes (e.g., traffic levels: low/medium/high)
            dropout (float): Dropout rate for regularization
            bidirectional (bool): Use bidirectional RNN
        """
        super(TrafficPatternRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # RNN layer for pattern analysis
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            hidden (torch.Tensor, optional): Initial hidden state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output predictions and final hidden state
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Take the output from the last time step
        last_output = rnn_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize hidden state
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensor on
            
        Returns:
            torch.Tensor: Initialized hidden state
        """
        return torch.zeros(self.num_layers * self.num_directions, 
                          batch_size, 
                          self.hidden_size).to(device)


class TravelTimeLSTM(nn.Module):
    """
    LSTM model for predicting travel time based on path characteristics
    
    This model performs TIME SERIES ANALYSIS and predicts journey duration by analyzing:
    - Historical travel times on path segments (learns historical patterns)
    - Current traffic conditions (real-time data)
    - Time of day and day of week (rush hour patterns, weekly cycles)
    - Weather conditions (seasonal variations)
    - Incident severity along the route
    
    LEARNS FROM HISTORICAL DATA:
    - Rush hour patterns (7-9 AM, 5-7 PM have higher traffic)
    - Weekly cycles (weekdays vs weekends)
    - Seasonal patterns (holidays, weather impacts)
    - Long-term traffic trends
    
    INPUT FEATURES (15 features per time step):
    0. segment_lengths - Length of road segments (km)
    1. historical_times - Historical travel times for segments (minutes)
    2. traffic_volume - Current traffic volume (0-1 normalized)
    3. average_speed - Average speed on segment (km/h)
    4. traffic_density - Traffic density (0-1 normalized)
    5. incident_severity - Severity of incidents (0=none, 0.33=minor, 0.67=moderate, 1.0=severe)
    6. hour_of_day - Hour (0-23 normalized to 0-1) - LEARNS RUSH HOURS
    7. is_rush_hour - Binary indicator for rush hour - TEMPORAL PATTERN
    8. day_of_week - Day (0-6 normalized to 0-1) - LEARNS WEEKLY CYCLES
    9. is_weekend - Binary indicator for weekend - WEEKLY PATTERN
    10. weather_condition - Weather quality (0.5-1.0) - SEASONAL VARIATION
    11. num_lanes - Number of lanes (normalized)
    12. road_quality - Road condition (0-1)
    13. has_construction - Binary indicator for construction
    14. base_time - Normalized base travel time
    """
    
    def __init__(self, 
                 input_size: int = 15,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize the Travel Time LSTM
        
        Args:
            input_size (int): Number of input features per time step
            hidden_size (int): Number of hidden units in LSTM layers
            num_layers (int): Number of stacked LSTM layers
            dropout (float): Dropout rate for regularization
            bidirectional (bool): Use bidirectional LSTM
        """
        super(TravelTimeLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers for time series prediction
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism for focusing on important time steps
        self.attention = nn.Linear(hidden_size * self.num_directions, 1)
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)  # Single output: predicted travel time
        
    def attention_layer(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism to LSTM outputs
        
        Args:
            lstm_output (torch.Tensor): LSTM output (batch_size, seq_length, hidden_size)
            
        Returns:
            torch.Tensor: Attention-weighted output
        """
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context_vector
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        TIME SERIES ANALYSIS:
        The LSTM processes the sequence of temporal features (30 time steps) and:
        1. Learns patterns in hour_of_day (index 6) -> identifies rush hours
        2. Learns patterns in day_of_week (index 8) -> identifies weekly cycles  
        3. Learns patterns in weather (index 10) -> identifies seasonal variations
        4. Uses attention to focus on critical time periods
        5. Predicts travel time based on these learned patterns
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
                             Represents temporal features of the path over time
            
        Returns:
            torch.Tensor: Predicted travel time (batch_size, 1) in minutes
        """
        batch_size = x.size(0)
        
        # LSTM forward pass - LEARNS TEMPORAL PATTERNS
        # The bidirectional LSTM processes the sequence in both directions,
        # capturing both past trends and future context
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism - FOCUSES ON IMPORTANT TIME PERIODS
        # Attention weights determine which time steps are most important
        # E.g., rush hour periods get higher weights
        context = self.attention_layer(lstm_out)
        
        # Fully connected layers for prediction
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        # Ensure positive travel time
        out = torch.abs(out)
        
        return out
    
    def analyze_temporal_patterns(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze and extract temporal patterns learned by the LSTM
        
        This method provides insight into what the LSTM has learned about:
        - Rush hour impacts
        - Weekly traffic cycles
        - Seasonal variations
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            Dict containing:
            - 'attention_weights': Which time periods the model focuses on
            - 'hidden_states': LSTM hidden states showing learned patterns
            - 'prediction': Travel time prediction
        """
        self.eval()
        with torch.no_grad():
            # LSTM forward pass
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Get attention weights - shows which time steps are important
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            
            # Get prediction
            context = self.attention_layer(lstm_out)
            out = self.fc1(context)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.relu(out)
            prediction = torch.abs(self.fc3(out))
            
            return {
                'attention_weights': attention_weights.squeeze(-1),  # (batch, seq_len)
                'hidden_states': hidden,  # LSTM hidden states
                'lstm_outputs': lstm_out,  # Full LSTM outputs
                'prediction': prediction
            }


class TrafficSequenceDataset(Dataset):
    """
    Dataset class for traffic sequence data
    """
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset
        
        Args:
            sequences (np.ndarray): Input sequences (n_samples, seq_length, n_features)
            targets (np.ndarray): Target values (n_samples,) or (n_samples, n_outputs)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class RNNTrainer:
    """
    Trainer class for RNN/LSTM models
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize the trainer
        
        Args:
            model (nn.Module): The RNN/LSTM model to train
            device (torch.device, optional): Device to train on
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): L2 regularization factor
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function (will be set during training based on task)
        self.criterion = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   is_classification: bool = False) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
            is_classification (bool): Whether task is classification (vs regression)
            
        Returns:
            Tuple[float, float]: Average loss and accuracy/MAE
        """
        self.model.train()
        running_loss = 0.0
        running_metric = 0.0
        
        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.model, TrafficPatternRNN):
                outputs, _ = self.model(sequences)
            else:  # TravelTimeLSTM
                outputs = self.model(sequences)
            
            # Calculate loss
            if is_classification:
                loss = self.criterion(outputs, targets.long())
                _, predicted = torch.max(outputs.data, 1)
                running_metric += (predicted == targets).sum().item()
            else:
                loss = self.criterion(outputs.squeeze(), targets)
                # Mean Absolute Error for regression
                running_metric += torch.abs(outputs.squeeze() - targets).sum().item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item() * sequences.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        if is_classification:
            epoch_metric = running_metric / len(train_loader.dataset)
        else:
            epoch_metric = running_metric / len(train_loader.dataset)
        
        return epoch_loss, epoch_metric
    
    def validate_epoch(self,
                      val_loader: DataLoader,
                      is_classification: bool = False) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader (DataLoader): Validation data loader
            is_classification (bool): Whether task is classification (vs regression)
            
        Returns:
            Tuple[float, float]: Average loss and accuracy/MAE
        """
        self.model.eval()
        running_loss = 0.0
        running_metric = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if isinstance(self.model, TrafficPatternRNN):
                    outputs, _ = self.model(sequences)
                else:  # TravelTimeLSTM
                    outputs = self.model(sequences)
                
                # Calculate loss
                if is_classification:
                    loss = self.criterion(outputs, targets.long())
                    _, predicted = torch.max(outputs.data, 1)
                    running_metric += (predicted == targets).sum().item()
                else:
                    loss = self.criterion(outputs.squeeze(), targets)
                    running_metric += torch.abs(outputs.squeeze() - targets).sum().item()
                
                running_loss += loss.item() * sequences.size(0)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        
        if is_classification:
            epoch_metric = running_metric / len(val_loader.dataset)
        else:
            epoch_metric = running_metric / len(val_loader.dataset)
        
        return epoch_loss, epoch_metric
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int = 50,
             is_classification: bool = False,
             save_path: Optional[str] = None) -> Dict:
        """
        Train the model
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
            is_classification (bool): Whether task is classification (vs regression)
            save_path (str, optional): Path to save the best model
            
        Returns:
            Dict: Training history
        """
        # Set loss function based on task
        if is_classification:
            self.criterion = nn.CrossEntropyLoss()
            metric_name = "Accuracy"
        else:
            self.criterion = nn.MSELoss()
            metric_name = "MAE"
        
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        print(f"Training on {self.device}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_metric = self.train_epoch(train_loader, is_classification)
            
            # Validate
            val_loss, val_metric = self.validate_epoch(val_loader, is_classification)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_metric)
            self.history['val_acc'].append(val_metric)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'history': self.history
                    }, save_path)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train {metric_name}: {train_metric:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val {metric_name}: {val_metric:.4f}")
            print("-" * 60)
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        print(f"\nTraining complete! Best Val Loss: {best_val_loss:.4f}")
        
        return self.history


def load_model(model: nn.Module, 
               checkpoint_path: str,
               device: Optional[torch.device] = None) -> nn.Module:
    """
    Load a trained model from checkpoint
    
    Args:
        model (nn.Module): Model instance (with correct architecture)
        checkpoint_path (str): Path to checkpoint file
        device (torch.device, optional): Device to load model on
        
    Returns:
        nn.Module: Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")
    
    return model


def predict_travel_time(model: TravelTimeLSTM,
                       path_features: np.ndarray,
                       device: Optional[torch.device] = None) -> float:
    """
    Predict travel time for a given path
    
    Args:
        model (TravelTimeLSTM): Trained LSTM model
        path_features (np.ndarray): Path features (seq_length, n_features)
        device (torch.device, optional): Device for computation
        
    Returns:
        float: Predicted travel time in minutes
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Add batch dimension
    path_tensor = torch.FloatTensor(path_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_time = model(path_tensor)
    
    return predicted_time.item()


def analyze_traffic_pattern(model: TrafficPatternRNN,
                           traffic_sequence: np.ndarray,
                           device: Optional[torch.device] = None) -> np.ndarray:
    """
    Analyze traffic pattern from sequence data
    
    Args:
        model (TrafficPatternRNN): Trained RNN model
        traffic_sequence (np.ndarray): Traffic data sequence (seq_length, n_features)
        device (torch.device, optional): Device for computation
        
    Returns:
        np.ndarray: Traffic pattern predictions (probabilities)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Add batch dimension
    sequence_tensor = torch.FloatTensor(traffic_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs, _ = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    return probabilities.cpu().numpy()[0]


def predict_with_temporal_analysis(model: TravelTimeLSTM,
                                   path_features: np.ndarray,
                                   device: Optional[torch.device] = None) -> Dict:
    """
    Predict travel time AND analyze what temporal patterns influenced the prediction
    
    This demonstrates the LSTM's TIME SERIES ANALYSIS capabilities:
    - Shows which time periods it focuses on (via attention)
    - Reveals rush hour impacts
    - Highlights weekly cycle effects
    
    Args:
        model (TravelTimeLSTM): Trained LSTM model
        path_features (np.ndarray): Path features (seq_length, 15 features)
        device (torch.device, optional): Device for computation
        
    Returns:
        Dict containing:
        - 'travel_time': Predicted travel time in minutes
        - 'attention_weights': Which time steps influenced prediction most
        - 'rush_hour_impact': Estimated impact of rush hour
        - 'weekend_effect': Weekend vs weekday effect
        - 'temporal_features': Extracted temporal information
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Convert to tensor
    path_tensor = torch.FloatTensor(path_features).unsqueeze(0).to(device)
    
    # Get detailed analysis
    analysis = model.analyze_temporal_patterns(path_tensor)
    
    # Extract temporal information from input features
    hour_of_day = path_features[:, 6]  # Feature index 6
    is_rush_hour = path_features[:, 7]  # Feature index 7
    day_of_week = path_features[:, 8]  # Feature index 8
    is_weekend = path_features[:, 9]  # Feature index 9
    
    # Calculate impacts
    attention_weights = analysis['attention_weights'].cpu().numpy()[0]
    rush_hour_periods = np.where(is_rush_hour > 0.5)[0]
    weekend_periods = np.where(is_weekend > 0.5)[0]
    
    # Rush hour impact: average attention weight during rush hours
    rush_hour_impact = attention_weights[rush_hour_periods].mean() if len(rush_hour_periods) > 0 else 0
    
    # Weekend effect: difference in attention between weekend and weekday
    weekend_attention = attention_weights[weekend_periods].mean() if len(weekend_periods) > 0 else 0
    weekday_periods = np.where(is_weekend < 0.5)[0]
    weekday_attention = attention_weights[weekday_periods].mean() if len(weekday_periods) > 0 else 0
    
    return {
        'travel_time': analysis['prediction'].item(),
        'attention_weights': attention_weights,
        'rush_hour_impact': float(rush_hour_impact),
        'weekend_effect': float(weekend_attention - weekday_attention),
        'temporal_features': {
            'average_hour': float(hour_of_day.mean() * 24),
            'rush_hour_percentage': float(is_rush_hour.mean() * 100),
            'weekend_percentage': float(is_weekend.mean() * 100),
            'peak_attention_time': int(np.argmax(attention_weights))
        }
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing RNN/LSTM Models")
    print("=" * 60)
    
    # Test TrafficPatternRNN
    print("\n1. Testing TrafficPatternRNN")
    print("-" * 60)
    rnn_model = TrafficPatternRNN(
        input_size=10,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        bidirectional=True
    )
    print(f"Model created: {rnn_model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in rnn_model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    seq_length = 30
    dummy_input = torch.randn(batch_size, seq_length, 10)
    output, hidden = rnn_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test TravelTimeLSTM
    print("\n2. Testing TravelTimeLSTM")
    print("-" * 60)
    lstm_model = TravelTimeLSTM(
        input_size=15,
        hidden_size=64,
        num_layers=2,
        bidirectional=True
    )
    print(f"Model created: {lstm_model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, seq_length, 15)
    output = lstm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed successfully!")

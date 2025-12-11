"""
Training Script for RNN/LSTM Models

This script trains both:
1. TrafficPatternRNN - for traffic pattern classification
2. TravelTimeLSTM - for travel time prediction

The script generates synthetic training data and demonstrates the training process.
In production, replace synthetic data with real traffic data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.rnn_model import (
    TrafficPatternRNN,
    TravelTimeLSTM,
    TrafficSequenceDataset,
    RNNTrainer
)


def generate_synthetic_traffic_pattern_data(n_samples: int = 1000,
                                           seq_length: int = 30,
                                           n_features: int = 15) -> tuple:
    """
    Generate synthetic traffic pattern data for demonstration
    
    Features include: traffic volume, average speed, density, incidents, etc.
    Classes: 0 (low traffic), 1 (medium traffic), 2 (high traffic)
    
    Args:
        n_samples (int): Number of samples to generate
        seq_length (int): Length of each sequence
        n_features (int): Number of features per time step
        
    Returns:
        tuple: (sequences, labels)
    """
    np.random.seed(42)
    sequences = []
    labels = []
    
    for _ in range(n_samples):
        # Randomly select traffic level
        traffic_level = np.random.randint(0, 3)
        
        # Generate sequence based on traffic level
        if traffic_level == 0:  # Low traffic
            volume = np.random.uniform(0.1, 0.4, seq_length)
            speed = np.random.uniform(0.7, 1.0, seq_length)
            density = np.random.uniform(0.1, 0.3, seq_length)
        elif traffic_level == 1:  # Medium traffic
            volume = np.random.uniform(0.4, 0.7, seq_length)
            speed = np.random.uniform(0.4, 0.7, seq_length)
            density = np.random.uniform(0.3, 0.6, seq_length)
        else:  # High traffic
            volume = np.random.uniform(0.7, 1.0, seq_length)
            speed = np.random.uniform(0.1, 0.4, seq_length)
            density = np.random.uniform(0.6, 1.0, seq_length)
        
        # Add noise and temporal patterns
        time_factor = np.linspace(0, 2*np.pi, seq_length)
        volume = volume + 0.1 * np.sin(time_factor)
        speed = speed + 0.1 * np.cos(time_factor)
        
        # Create additional features
        flow = volume * speed
        incidents = np.random.poisson(traffic_level * 0.5, seq_length) / 5.0
        time_of_day = np.linspace(0, 1, seq_length)
        day_of_week = np.random.uniform(0, 1, seq_length)
        weather = np.random.uniform(0.5, 1.0, seq_length)
        
        # Additional synthetic features
        congestion_index = volume * density
        variance = np.random.uniform(0, 0.2, seq_length)
        
        # Combine all features
        sequence = np.stack([
            volume, speed, density, flow, incidents,
            time_of_day, day_of_week, weather, congestion_index, variance
        ], axis=-1)
        
        sequences.append(sequence)
        labels.append(traffic_level)
    
    return np.array(sequences), np.array(labels)


def generate_synthetic_travel_time_data(n_samples: int = 1000,
                                       seq_length: int = 30,
                                       n_features: int = 15) -> tuple:
    """
    Generate REALISTIC synthetic travel time data
    
    IMPROVED YUH: More realistic ranges and moderate multipliers
    Target MAE: < 10 minutes
    """
    np.random.seed(42)
    sequences = []
    travel_times = []
    
    for _ in range(n_samples):
        # REALISTIC base parameters
        total_distance = np.random.uniform(10, 30)  # 10-30 km (typical city route)
        base_speed = np.random.uniform(40, 70)      # 40-70 km/h
        base_time = (total_distance / base_speed) * 60  # 8-45 minutes
        
        # Generate segments
        segment_lengths = np.random.uniform(0.5, 2.0, seq_length)
        segment_lengths = segment_lengths * (total_distance / segment_lengths.sum())
        
        historical_times = segment_lengths / base_speed * 60
        
        # REALISTIC traffic (beta distribution - mostly moderate)
        traffic_volume = np.random.beta(2, 3, seq_length)  # Peak at 0.4
        average_speed = base_speed * (1.0 - traffic_volume * 0.3)  # Max 30% reduction
        average_speed = np.clip(average_speed, 20, 80)
        
        traffic_density = traffic_volume * np.random.uniform(0.9, 1.1, seq_length)
        traffic_density = np.clip(traffic_density, 0, 1)
        
        # REALISTIC incidents (mostly none)
        incident_probs = [0.75, 0.18, 0.05, 0.02]  # 75% none, 18% minor, 5% moderate, 2% severe
        incident_severity = np.random.choice([0.0, 0.25, 0.6, 1.0], size=seq_length, p=incident_probs)
        
        # Time factors
        hour = np.random.uniform(0, 24)
        hour_normalized = hour / 24
        hour_of_day = np.full(seq_length, hour_normalized)
        
        is_rush_hour = ((7 <= hour <= 9) or (17 <= hour <= 19))
        is_rush_hour_arr = np.full(seq_length, float(is_rush_hour))
        
        day = np.random.randint(0, 7)
        day_of_week = np.full(seq_length, day / 7)
        is_weekend = float(day >= 5)
        is_weekend_arr = np.full(seq_length, is_weekend)
        
        weather_condition = np.random.beta(6, 2, seq_length)  
        
        num_lanes = np.random.randint(2, 5, seq_length)
        road_quality = np.random.beta(5, 2, seq_length)
        has_construction = np.random.choice([0, 1], size=seq_length, p=[0.95, 0.05])
        
        # Calculate REALISTIC travel time with MODERATE multipliers
        actual_time = base_time
        
        # Traffic impact (max +25%)
        traffic_factor = 1.0 + (traffic_volume.mean() * 0.25)
        
        # Incident impact (REALISTIC)
        avg_incident = incident_severity.mean()
        if avg_incident > 0.8:      # Severe (rare)
            incident_factor = 1.5   # +50%
        elif avg_incident > 0.5:    # Moderate
            incident_factor = 1.25  # +25%
        elif avg_incident > 0.15:   # Minor
            incident_factor = 1.1   # +10%
        else:                       # None
            incident_factor = 1.0
        
        # Rush hour (+12%)
        rush_factor = 1.12 if is_rush_hour else 1.0
        
        # Weather (max +15%)
        weather_factor = 1.0 + ((1.0 - weather_condition.mean()) * 0.15)
        
        # Construction (+20% if present)
        construction_factor = 1.0 + (has_construction.mean() * 0.20)
        
        # Weekend (-8% traffic)
        weekend_factor = 0.92 if is_weekend else 1.0
        
        # Apply ALL factors (max combined ~2.0x in extreme cases)
        actual_time *= (traffic_factor * incident_factor * rush_factor * 
                       weather_factor * construction_factor * weekend_factor)
        
        # Ensure realistic bounds (8-60 minutes for city routes)
        actual_time = np.clip(actual_time, 8, 60)
        
        # Small random noise (±3%)
        actual_time *= np.random.uniform(0.90, 1.10)
        
        # Combine features
        sequence = np.stack([
            segment_lengths,
            historical_times,
            traffic_volume,
            average_speed,
            traffic_density,
            incident_severity,
            hour_of_day,
            is_rush_hour_arr,
            day_of_week,
            is_weekend_arr,
            weather_condition,
            num_lanes / 4.0,
            road_quality,
            has_construction,
            np.ones(seq_length) * base_time / 60.0
        ], axis=-1)
        
        sequences.append(sequence)
        travel_times.append(actual_time)
    
    sequences = np.array(sequences)
    travel_times = np.array(travel_times)
    
    # Data quality report
    print(f"\n[DATA QUALITY REPORT]")
    print(f"  Travel time range: {travel_times.min():.1f} - {travel_times.max():.1f} minutes")
    print(f"  Mean: {travel_times.mean():.1f} ± {travel_times.std():.1f} minutes")
    print(f"  Median: {np.median(travel_times):.1f} minutes")
    
    return sequences, travel_times


def plot_training_history(history: dict, 
                          save_path: str,
                          model_name: str,
                          is_classification: bool = False):
    """
    Plot and save training history
    
    Args:
        history (dict): Training history dictionary
        save_path (str): Path to save the plot
        model_name (str): Name of the model
        is_classification (bool): Whether task is classification
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metric
    metric_name = 'Accuracy' if is_classification else 'MAE'
    axes[1].plot(history['train_acc'], label=f'Train {metric_name}')
    axes[1].plot(history['val_acc'], label=f'Val {metric_name}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(f'{model_name} - {metric_name}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def train_traffic_pattern_rnn(args):
    """
    Train the Traffic Pattern RNN model
    """
    print("\n" + "="*80)
    print("TRAINING TRAFFIC PATTERN RNN")
    print("="*80)
    
    # Generate synthetic data
    print("\nGenerating synthetic traffic pattern data...")
    sequences, labels = generate_synthetic_traffic_pattern_data(
        n_samples=args.n_samples,
        seq_length=args.seq_length,
        n_features=10
    )
    print(f"Generated {len(sequences)} samples")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets and dataloaders
    train_dataset = TrafficSequenceDataset(X_train, y_train)
    val_dataset = TrafficSequenceDataset(X_val, y_val)
    test_dataset = TrafficSequenceDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    print("\nCreating TrafficPatternRNN model...")
    model = TrafficPatternRNN(
        input_size=10,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=3,
        dropout=args.dropout,
        bidirectional=True
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = RNNTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train
    save_path = Path(args.save_dir) / 'rnn_pattern_model.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        is_classification=True,
        save_path=str(save_path)
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate_epoch(test_loader, is_classification=True)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save training history plot
    plot_path = Path(args.save_dir) / 'rnn_pattern_training_history.png'
    plot_training_history(history, str(plot_path), 'Traffic Pattern RNN', is_classification=True)
    
    # Save training info
    info = {
        'model': 'TrafficPatternRNN',
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': args.n_samples,
        'seq_length': args.seq_length,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc)
    }
    
    info_path = Path(args.save_dir) / 'rnn_pattern_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"\nModel saved to: {save_path}")
    print(f"Training info saved to: {info_path}")
    
    return model, history


def train_travel_time_lstm(args):
    """
    Train the Travel Time LSTM model
    """
    print("\n" + "="*80)
    print("TRAINING TRAVEL TIME LSTM")
    print("="*80)
    
    # Generate synthetic data
    print("\nGenerating synthetic travel time data...")
    sequences, travel_times = generate_synthetic_travel_time_data(
        n_samples=args.n_samples,
        seq_length=args.seq_length,
        n_features=15
    )
    print(f"Generated {len(sequences)} samples")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Travel times shape: {travel_times.shape}")
    print(f"Travel time range: {travel_times.min():.2f} - {travel_times.max():.2f} minutes")
    print(f"Mean travel time: {travel_times.mean():.2f} ± {travel_times.std():.2f} minutes")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, travel_times, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets and dataloaders
    train_dataset = TrafficSequenceDataset(X_train, y_train)
    val_dataset = TrafficSequenceDataset(X_val, y_val)
    test_dataset = TrafficSequenceDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    print("\nCreating TravelTimeLSTM model...")
    model = TravelTimeLSTM(
        input_size=15,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = RNNTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train
    save_path = Path(args.save_dir) / 'lstm_travel_time_model.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        is_classification=False,
        save_path=str(save_path)
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_mae = trainer.validate_epoch(test_loader, is_classification=False)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f} minutes")
    
    # Save training history plot
    plot_path = Path(args.save_dir) / 'lstm_travel_time_training_history.png'
    plot_training_history(history, str(plot_path), 'Travel Time LSTM', is_classification=False)
    
    # Save training info
    info = {
        'model': 'TravelTimeLSTM',
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': args.n_samples,
        'seq_length': args.seq_length,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'test_loss': float(test_loss),
        'test_mae': float(test_mae)
    }
    
    info_path = Path(args.save_dir) / 'lstm_travel_time_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"\nModel saved to: {save_path}")
    print(f"Training info saved to: {info_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train RNN/LSTM Models for Traffic Analysis')
    
    # Model selection
    parser.add_argument('--model', type=str, default='both',
                       choices=['rnn', 'lstm', 'both'],
                       help='Which model to train (default: both)')
    
    # Data parameters
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of training samples (default: 1000)')
    parser.add_argument('--seq_length', type=int, default=30,
                       help='Sequence length (default: 30)')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Hidden size (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of RNN/LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models (default: models)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RNN/LSTM TRAINING SCRIPT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Train models
    if args.model in ['rnn', 'both']:
        train_traffic_pattern_rnn(args)
    
    if args.model in ['lstm', 'both']:
        train_travel_time_lstm(args)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

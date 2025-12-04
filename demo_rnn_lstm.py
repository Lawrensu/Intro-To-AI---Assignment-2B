"""
Demo Script for RNN/LSTM Models

This script demonstrates how to:
1. Use the trained TrafficPatternRNN for traffic pattern analysis
2. Use the trained TravelTimeLSTM for travel time prediction
3. Integrate predictions with pathfinding

Run this after training the models with train_lstm.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.rnn_model import (
    TrafficPatternRNN,
    TravelTimeLSTM,
    load_model,
    predict_travel_time,
    analyze_traffic_pattern
)


def demo_traffic_pattern_analysis():
    """
    Demonstrate traffic pattern analysis using RNN
    """
    print("\n" + "="*80)
    print("DEMO: TRAFFIC PATTERN ANALYSIS")
    print("="*80)
    
    # Check if model exists
    model_path = Path('models') / 'rnn_pattern_model.pth'
    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("Please train the model first using:")
        print("  python src/train_lstm.py --model rnn")
        return
    
    # Load the trained model
    print("\nLoading trained Traffic Pattern RNN...")
    model = TrafficPatternRNN(
        input_size=10,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        bidirectional=True
    )
    model = load_model(model, str(model_path))
    
    # Create sample traffic sequence (representing 30 time steps)
    print("\nAnalyzing sample traffic patterns...")
    
    # Scenario 1: Morning rush hour - High traffic
    print("\n1. Morning Rush Hour (High Traffic Expected)")
    print("-" * 60)
    sequence1 = np.random.rand(30, 10)
    sequence1[:, 0] = 0.8  # High volume
    sequence1[:, 1] = 0.3  # Low speed
    sequence1[:, 2] = 0.7  # High density
    sequence1[:, 5] = 0.3  # Morning time (7-9 AM)
    
    probabilities1 = analyze_traffic_pattern(model, sequence1)
    print(f"Traffic Pattern Probabilities:")
    print(f"  Low Traffic:    {probabilities1[0]:.2%}")
    print(f"  Medium Traffic: {probabilities1[1]:.2%}")
    print(f"  High Traffic:   {probabilities1[2]:.2%}")
    print(f"Prediction: {'High' if probabilities1[2] > 0.5 else 'Medium' if probabilities1[1] > 0.5 else 'Low'} Traffic")
    
    # Scenario 2: Late night - Low traffic
    print("\n2. Late Night (Low Traffic Expected)")
    print("-" * 60)
    sequence2 = np.random.rand(30, 10)
    sequence2[:, 0] = 0.2  # Low volume
    sequence2[:, 1] = 0.9  # High speed
    sequence2[:, 2] = 0.2  # Low density
    sequence2[:, 5] = 0.9  # Night time (10 PM - 12 AM)
    
    probabilities2 = analyze_traffic_pattern(model, sequence2)
    print(f"Traffic Pattern Probabilities:")
    print(f"  Low Traffic:    {probabilities2[0]:.2%}")
    print(f"  Medium Traffic: {probabilities2[1]:.2%}")
    print(f"  High Traffic:   {probabilities2[2]:.2%}")
    print(f"Prediction: {'High' if probabilities2[2] > 0.5 else 'Medium' if probabilities2[1] > 0.5 else 'Low'} Traffic")
    
    # Scenario 3: Midday - Medium traffic
    print("\n3. Midday (Medium Traffic Expected)")
    print("-" * 60)
    sequence3 = np.random.rand(30, 10)
    sequence3[:, 0] = 0.5  # Medium volume
    sequence3[:, 1] = 0.6  # Medium speed
    sequence3[:, 2] = 0.5  # Medium density
    sequence3[:, 5] = 0.5  # Midday time (12 PM - 2 PM)
    
    probabilities3 = analyze_traffic_pattern(model, sequence3)
    print(f"Traffic Pattern Probabilities:")
    print(f"  Low Traffic:    {probabilities3[0]:.2%}")
    print(f"  Medium Traffic: {probabilities3[1]:.2%}")
    print(f"  High Traffic:   {probabilities3[2]:.2%}")
    print(f"Prediction: {'High' if probabilities3[2] > 0.5 else 'Medium' if probabilities3[1] > 0.5 else 'Low'} Traffic")


def demo_travel_time_prediction():
    """
    Demonstrate travel time prediction using LSTM
    """
    print("\n" + "="*80)
    print("DEMO: TRAVEL TIME PREDICTION")
    print("="*80)
    
    # Check if model exists
    model_path = Path('models') / 'lstm_travel_time_model.pth'
    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("Please train the model first using:")
        print("  python src/train_lstm.py --model lstm")
        return
    
    # Load the trained model
    print("\nLoading trained Travel Time LSTM...")
    model = TravelTimeLSTM(
        input_size=15,
        hidden_size=64,
        num_layers=2,
        bidirectional=True
    )
    model = load_model(model, str(model_path))
    
    # Create sample path features (30 segments)
    print("\nPredicting travel times for different scenarios...")
    
    # Scenario 1: Clear route, no incidents
    print("\n1. Clear Route - No Incidents")
    print("-" * 60)
    path1 = np.random.rand(30, 15)
    path1[:, 0] = 2.0  # Segment lengths (km)
    path1[:, 1] = 5.0  # Historical time (minutes)
    path1[:, 2] = 0.3  # Low traffic volume
    path1[:, 3] = 60.0  # Good speed (km/h)
    path1[:, 5] = 0.0  # No incidents
    path1[:, 10] = 0.9  # Good weather
    
    time1 = predict_travel_time(model, path1)
    print(f"Predicted Travel Time: {time1:.2f} minutes")
    total_distance = np.sum(path1[:, 0])
    print(f"Total Distance: {total_distance:.2f} km")
    print(f"Average Speed: {(total_distance / time1 * 60):.2f} km/h")
    
    # Scenario 2: Heavy traffic with moderate incident
    print("\n2. Heavy Traffic with Moderate Incident")
    print("-" * 60)
    path2 = np.random.rand(30, 15)
    path2[:, 0] = 2.0  # Segment lengths
    path2[:, 1] = 8.0  # Historical time (higher)
    path2[:, 2] = 0.8  # High traffic volume
    path2[:, 3] = 30.0  # Slow speed
    path2[:, 5] = 0.67  # Moderate incident
    path2[:, 7] = 1.0  # Rush hour
    path2[:, 10] = 0.7  # Fair weather
    
    time2 = predict_travel_time(model, path2)
    print(f"Predicted Travel Time: {time2:.2f} minutes")
    total_distance = np.sum(path2[:, 0])
    print(f"Total Distance: {total_distance:.2f} km")
    print(f"Average Speed: {(total_distance / time2 * 60):.2f} km/h")
    print(f"Time increase vs. clear route: {((time2 - time1) / time1 * 100):.1f}%")
    
    # Scenario 3: Severe incident with bad weather
    print("\n3. Severe Incident with Bad Weather")
    print("-" * 60)
    path3 = np.random.rand(30, 15)
    path3[:, 0] = 2.0  # Segment lengths
    path3[:, 1] = 10.0  # Historical time (much higher)
    path3[:, 2] = 0.9  # Very high traffic
    path3[:, 3] = 20.0  # Very slow speed
    path3[:, 5] = 1.0  # Severe incident
    path3[:, 10] = 0.5  # Poor weather
    path3[:, 13] = 1.0  # Construction
    
    time3 = predict_travel_time(model, path3)
    print(f"Predicted Travel Time: {time3:.2f} minutes")
    total_distance = np.sum(path3[:, 0])
    print(f"Total Distance: {total_distance:.2f} km")
    print(f"Average Speed: {(total_distance / time3 * 60):.2f} km/h")
    print(f"Time increase vs. clear route: {((time3 - time1) / time1 * 100):.1f}%")


def demo_integrated_route_planning():
    """
    Demonstrate integrated route planning with RNN/LSTM predictions
    """
    print("\n" + "="*80)
    print("DEMO: INTEGRATED ROUTE PLANNING")
    print("="*80)
    
    print("\nScenario: Planning optimal route from Heritage to Sentral")
    print("-" * 60)
    
    # Simulate multiple route options
    routes = [
        {
            'name': 'Route A (Highway)',
            'distance': 15.5,
            'segments': 20,
            'traffic_level': 'low',
            'incidents': 0
        },
        {
            'name': 'Route B (City Center)',
            'distance': 12.3,
            'segments': 25,
            'traffic_level': 'high',
            'incidents': 1
        },
        {
            'name': 'Route C (Bypass)',
            'distance': 18.2,
            'segments': 18,
            'traffic_level': 'medium',
            'incidents': 0
        }
    ]
    
    print("\nAvailable Routes:")
    for i, route in enumerate(routes, 1):
        print(f"\n{i}. {route['name']}")
        print(f"   Distance: {route['distance']} km")
        print(f"   Segments: {route['segments']}")
        print(f"   Traffic Level: {route['traffic_level']}")
        print(f"   Active Incidents: {route['incidents']}")
    
    print("\n" + "-" * 60)
    print("Analyzing routes with ML predictions...")
    print("-" * 60)
    
    # Check if models exist
    rnn_path = Path('models') / 'rnn_pattern_model.pth'
    lstm_path = Path('models') / 'lstm_travel_time_model.pth'
    
    if not rnn_path.exists() or not lstm_path.exists():
        print("\nModels not found. Showing estimated times instead.")
        print("\nRoute Comparison (Estimated):")
        for i, route in enumerate(routes, 1):
            base_time = route['distance'] / 50 * 60  # Base at 50 km/h
            multiplier = {'low': 1.0, 'medium': 1.3, 'high': 1.6}[route['traffic_level']]
            incident_penalty = route['incidents'] * 10
            total_time = base_time * multiplier + incident_penalty
            print(f"\n{route['name']}: {total_time:.1f} minutes")
    else:
        # Load models
        print("\nLoading trained models...")
        rnn_model = TrafficPatternRNN(input_size=10, hidden_size=64, num_layers=2, output_size=3, bidirectional=True)
        rnn_model = load_model(rnn_model, str(rnn_path))
        
        lstm_model = TravelTimeLSTM(input_size=15, hidden_size=64, num_layers=2, bidirectional=True)
        lstm_model = load_model(lstm_model, str(lstm_path))
        
        print("\nRoute Comparison (ML Predictions):")
        predictions = []
        
        for route in routes:
            # Generate synthetic path features
            seq_len = route['segments']
            path_features = np.random.rand(seq_len, 15)
            
            # Set relevant features based on route characteristics
            path_features[:, 0] = route['distance'] / seq_len  # Segment length
            path_features[:, 2] = {'low': 0.3, 'medium': 0.6, 'high': 0.9}[route['traffic_level']]
            path_features[:, 5] = route['incidents'] * 0.67  # Incident severity
            
            # Predict travel time
            predicted_time = predict_travel_time(lstm_model, path_features)
            predictions.append((route['name'], predicted_time, route['distance']))
        
        # Sort by predicted time
        predictions.sort(key=lambda x: x[1])
        
        print("\nRecommended Route Order:")
        for i, (name, time, distance) in enumerate(predictions, 1):
            avg_speed = (distance / time * 60)
            print(f"{i}. {name}")
            print(f"   Predicted Time: {time:.1f} minutes")
            print(f"   Distance: {distance} km")
            print(f"   Average Speed: {avg_speed:.1f} km/h")
        
        print(f"\n✓ Recommended: {predictions[0][0]} ({predictions[0][1]:.1f} minutes)")


def main():
    """
    Main demo function
    """
    print("="*80)
    print("RNN/LSTM MODELS DEMONSTRATION")
    print("="*80)
    print("\nThis demo shows the capabilities of:")
    print("  1. TrafficPatternRNN - Analyzes traffic patterns")
    print("  2. TravelTimeLSTM - Predicts journey travel times")
    print("  3. Integrated Route Planning - Combines both for optimal routing")
    
    # Run demos
    demo_traffic_pattern_analysis()
    demo_travel_time_prediction()
    demo_integrated_route_planning()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  - Train models: python src/train_lstm.py --model both")
    print("  - Integrate with pathfinding: Modify src/pathfinding.py")
    print("  - Add to GUI: Update src/gui.py with RNN/LSTM predictions")
    print("\nFor more information, see src/models/rnn_model.py")


if __name__ == "__main__":
    main()

# Models Directory

This directory stores trained model checkpoints and training outputs.

## Contents

After training, you will find:

- `cnn_model.pth` - Best CNN model checkpoint (ResNet-18)
- `cnn_training_history.png` - Training/validation curves
- `cnn_confusion_matrix.png` - Confusion matrix (3x3)
- `rnn_pattern_model.pth` - Traffic Pattern RNN checkpoint
- `rnn_pattern_training_history.png` - RNN training curves
- `rnn_pattern_info.json` - RNN model configuration and metrics
- `lstm_travel_time_model.pth` - Travel Time LSTM checkpoint
- `lstm_travel_time_training_history.png` - LSTM training curves
- `lstm_travel_time_info.json` - LSTM model configuration and metrics
- `gcn_model.pth` - GCN model checkpoint

## Note

**Model files (.pth, .pt) are NOT tracked by Git** due to their large size (>100MB).

To get the trained models:
1. Train them yourself using the training scripts
2. Or download pre-trained models from the shared drive (if available)

## Training Models

```bash
# Train CNN model
python src/train_cnn.py

# Train LSTM model 
python src/train_lstm.py

# Train GCN model 
python src/train_gcn.py
```

## Model Specifications

### CNN (ResNet-18)
- **Size**: ~45 MB
- **Classes**: 3 (Minor, Moderate, Severe)
- **Input**: 224x224 RGB images
- **Output**: 3-class probabilities

### RNN (Traffic Pattern Analysis)
- **Size**: ~3 MB
- **Architecture**: Bidirectional RNN with 2 layers
- **Hidden Size**: 64
- **Input**: Time-series traffic sequences (seq_length × 10 features)
- **Output**: 3-class traffic pattern (low/medium/high)
- **Features**: traffic volume, speed, density, flow, incidents, time factors

### LSTM (Travel Time Prediction)
- **Size**: ~5 MB
- **Architecture**: Bidirectional LSTM with 2 layers + Attention
- **Hidden Size**: 64
- **Input**: Path segment sequences (seq_length × 15 features)
- **Output**: Predicted travel time (minutes)
- **Features**: segment data, traffic conditions, incidents, temporal factors

### GCN
- **Size**: ~3 MB
- **Input**: Road network graph
- **Output**: Traffic flow predictions
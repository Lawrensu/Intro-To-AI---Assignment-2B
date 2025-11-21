# Models Directory

This directory stores trained model checkpoints and training outputs.

## Contents

After training, you will find:

- `cnn_model.pth` - Best CNN model checkpoint (ResNet-18)
- `cnn_training_history.png` - Training/validation curves
- `cnn_confusion_matrix.png` - Confusion matrix (3x3)
- `lstm_model.pth` - LSTM model checkpoint
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

### LSTM
- **Size**: ~5 MB
- **Input**: Time-series traffic data
- **Output**: Travel time predictions

### GCN
- **Size**: ~3 MB
- **Input**: Road network graph
- **Output**: Traffic flow predictions
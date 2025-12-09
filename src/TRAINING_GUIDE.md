# Training Guide 

**Guide to training all 3 models**

---

## Before You Start


**Check dataset:**
```bash
python src/data_processing.py
# Should show: 4200-5600 train images
```

---

## 1️Train CNN Model (Incident Classification)

### **What it does:**
Analyzes incident images → Predicts severity (None/Minor/Moderate/Severe)

### **Training Command:**
```bash
python src/train_cnn.py
```

### **Configuration:**
```python
# Located in src/train_cnn.py
BATCH_SIZE = 32        # Reduce to 16 if GPU memory issues
NUM_EPOCHS = 25        # Training cycles
LEARNING_RATE = 0.001  # Adam optimizer learning rate
```

### **Expected Output:**
```
Training CNN model on cuda...
Classes: ['minor', 'moderate', 'none', 'severe']
Training samples: 4200

Epoch 1/25 (45.2s)
  Train Loss: 1.234 | Train Acc: 0.523
  Val Loss: 1.012 | Val Acc: 0.612
  *** New best model! (Val Acc: 0.612) ***

Epoch 2/25 (43.8s)
  Train Loss: 0.987 | Train Acc: 0.678
  Val Loss: 0.876 | Val Acc: 0.723
  *** New best model! (Val Acc: 0.723) ***
...

Training complete in 18.5 minutes
Best validation accuracy: 0.925

[RESULTS] Test Set Performance:
  Overall Accuracy: 0.918
  Precision: 0.915
  Recall: 0.918
  F1-Score: 0.916

[PER-CLASS] Performance:
  NONE:
    Precision: 0.942
    Recall: 0.951
    F1-Score: 0.946
  MINOR:
    Precision: 0.891
    Recall: 0.903
    F1-Score: 0.897
  MODERATE:
    Precision: 0.901
    Recall: 0.896
    F1-Score: 0.898
  SEVERE:
    Precision: 0.923
    Recall: 0.929
    F1-Score: 0.926
```

### **Output Files:**
- `models/cnn_model.pth` (~45 MB) - Best model checkpoint
- `models/cnn_training_history.png` - Loss/accuracy curves
- `models/cnn_confusion_matrix.png` - 4×4 confusion matrix

### **What to Check:**
Validation accuracy > 90%  
No large gap between train/val accuracy (<5%)  
Confusion matrix shows good predictions on diagonal  
Training completes without errors  

---

## Train LSTM/RNN Models (Time Series Prediction)

### **What it does:**
- **RNN**: Analyzes traffic patterns → Predicts traffic level (Low/Medium/High)
- **LSTM**: Analyzes route features → Predicts travel time adjustments

### **Training Command:**
```bash
# Train both models
python src/train_lstm.py --model both

# Or train separately:
python src/train_lstm.py --model rnn    # Traffic pattern only
python src/train_lstm.py --model lstm   # Travel time only
```

### **Configuration:**
```python
# Command line arguments:
--n_samples 2000      # Number of training samples
--epochs 30           # Training epochs
--batch_size 32       # Batch size
--hidden_size 64      # LSTM hidden units
--num_layers 2        # LSTM layers
```

### **Expected Output:**

**RNN Training:**
```
TRAINING TRAFFIC PATTERN RNN
========================================

Generating synthetic traffic pattern data...
Generated 2000 samples
Class distribution: [667 666 667]

Creating model...
Total parameters: 89,539

Starting training...
Epoch 1/30 (2.3s)
  Train Loss: 0.987 | Train Acc: 0.523
  Val Loss: 0.876 | Val Acc: 0.612

...

Training complete in 5.2 minutes
Best validation accuracy: 0.887

Test Results:
  Accuracy: 0.881
  Precision: 0.879
  Recall: 0.881
  F1-Score: 0.880
```

**LSTM Training:**
```
TRAINING TRAVEL TIME LSTM
========================================

Generating synthetic travel time data...
Generated 2000 samples

Creating model...
Total parameters: 156,737

Starting training...
Epoch 1/30 (2.8s)
  Train Loss: 45.23 | Train MAE: 12.45
  Val Loss: 38.76 | Val MAE: 10.23

...

Training complete in 6.1 minutes
Best validation MAE: 8.34

Test Results:
  MAE: 8.67 minutes
  RMSE: 11.23 minutes
  R²: 0.824
```

### **Output Files:**
- `models/rnn_pattern_model.pth` (~5 MB)
- `models/lstm_travel_time_model.pth` (~5 MB)
- `models/rnn_pattern_training_history.png`
- `models/lstm_travel_time_training_history.png`

### **What to Check:**
RNN accuracy > 85%  
LSTM MAE < 10 minutes  
Training curves show smooth convergence  
No overfitting (train/val gap small)  

---

## Train GCN Model (Road Network Analysis)

### **What it does:**
Analyzes road network graph → Predicts traffic flow (Low/Medium/High) at each intersection

### **Training Command:**
```bash
python src/train_gcn.py
```

### **Configuration:**
```python
# Located in src/train_gcn.py
NUM_EPOCHS = 200      # Training epochs
LEARNING_RATE = 0.01  # Learning rate
PATIENCE = 20         # Early stopping patience
```

### **Expected Output:**
```
GCN Model Training for Traffic Flow Prediction
================================================

Step 1: Loading road network...
✓ Loaded 1435 nodes, 3028 edges

Step 2: Creating GCN model...
✓ Model created successfully
  Total parameters: 45,123
  Trainable parameters: 45,123

Step 3: Training model...

Epoch 1/200 (1.2s)
  Train Loss: 0.987 | Train Acc: 0.456
  Val Loss: 0.912 | Val Acc: 0.523
  *** New best model! ***

...

Epoch 45/200 (1.1s)
  Train Loss: 0.234 | Train Acc: 0.923
  Val Loss: 0.278 | Val Acc: 0.887

Early stopping at epoch 45
Best validation accuracy: 0.892

Step 4: Evaluating on test set...
Test Results:
  Loss: 0.289
  Accuracy: 0.879

✓ Best model saved to: models/gcn_model.pth
✓ Training history: models/gcn_training_history.png
✓ Confusion matrix: models/gcn_confusion_matrix.png
```

### **Output Files:**
- `models/gcn_model.pth` (~3 MB)
- `models/gcn_training_history.png`
- `models/gcn_confusion_matrix.png`

### **What to Check:**
Test accuracy > 85%  
Early stopping triggers (indicates good convergence)  
Loss decreases smoothly  
Confusion matrix shows balanced predictions  

---

## Verify All Models Work

### **Test CNN:**
```python
from src.models.cnn_model import load_cnn_model
import torch

model = load_cnn_model('models/cnn_model.pth')
print("✓ CNN loaded successfully")
```

### **Test LSTM:**
```bash
python demo_rnn_lstm.py
```

Expected output shows traffic predictions and travel times.

### **Test Integration:**
```bash
python src/integration.py
```

This tests all 3 models working together.

---

## Understanding Results

### **Good Training Signs:**
Loss decreases over epochs  
Accuracy increases over epochs  
Val accuracy close to train accuracy (<5% gap)  
Confusion matrix has high values on diagonal  
No warnings or errors during training  

### **Bad Training Signs:**
❌ Loss increases or fluctuates wildly  
❌ Val accuracy much lower than train (overfitting)  
❌ Accuracy stuck at ~25% (random guessing for 4 classes)  
❌ Confusion matrix shows random predictions  
❌ "CUDA out of memory" errors  

---

## Troubleshooting Training Issues

### **Problem: "CUDA out of memory"**
```python
# Reduce batch size in training script:
BATCH_SIZE = 16  # or 8
```

### **Problem: Training is very slow (CPU)**
**Options:**
1. Use Google Colab (see `COLAB_GUIDE.md`)
2. Reduce epochs: `--epochs 10`
3. Reduce samples: `--n_samples 1000`
4. Wait (2-4 hours on CPU vs 20 minutes on GPU)

### **Problem: "Dataset not sufficient"**
```bash
# Re-download dataset
python kaggleDataset.py
```

### **Problem: Model accuracy too low (<80%)**
**Possible causes:**
- Dataset too small → Add more images
- Too few epochs → Increase to 50
- Learning rate too high → Reduce to 0.0005
- No data augmentation → Check `data_processing.py`

### **Problem: Overfitting (train acc >> val acc)**
**Solutions:**
- Add dropout: `dropout=0.5`
- Use data augmentation (already enabled)
- Reduce model complexity
- Add more training data

---
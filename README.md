# Traffic Incident Severity Classification System 

**COS30019 - Introduction to AI - Assignment 2B**

A machine learning system that predicts traffic incident severity and provides optimal route recommendations for Kuching, Malaysia.

---

## What Does This System Do?

1. **Analyzes accident images** → Predicts severity (None/Minor/Moderate/Severe)
2. **Uses 3 AI models**: CNN (images) + LSTM (time patterns) + GCN (road networks)
3. **Calculates travel times** → Adjusts routes based on incident severity
4. **Shows results on map** → Interactive GUI with route visualization

---

## Team Members

- **Lawrence Lian** (Leader) - CNN Model & Integration
- **Mohd Faridz** - LSTM Model
- **Cherrylynn** - GCN Model  
- **Jason Hernando** - GUI & Visualization

---

## Quick Start (3 Steps)

### **Step 1: Setup Environment (5 minutes)**

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Intro-To-AI---Assignment-2B

# 2. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# 3. Install PyTorch (choose ONE based on your GPU)
# For NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install other packages
pip install -r requirements.txt
```

### **Step 2: Download Dataset (10-30 minutes)**

```bash
python kaggleDataset.py
```

**What it does:**
- Downloads 5 datasets from Kaggle (~6-8 GB)
- Organizes into 4 classes (None, Minor, Moderate, Severe)
- Splits into train/val/test (70%/15%/15%)
- **Result**: 6,000-8,000 images ready for training

### **Step 3: Train Models (30-60 minutes total)**

```bash
# Train CNN (20-30 minutes)
python src/train_cnn.py

# Train LSTM (5-10 minutes)
python src/train_lstm.py --model both

# Train GCN (10-15 minutes)
python src/train_gcn.py
```

**Done!** Models are saved in [`models/`](models/) folder.

---

## System Architecture

### **4-Class Classification System**

| Class | Description | Time Multiplier |
|-------|-------------|-----------------|
| **None** | No damage | 1.0× (no delay) |
| **Minor** | Light damage | 1.2× (+20% time) |
| **Moderate** | Moderate damage | 1.5× (+50% time) |
| **Severe** | Heavy damage | 2.0× (+100% time) |

### **Model Overview**

```
INPUT (Incident Image) → CNN Model → Severity Prediction
                                          ↓
Road Network → GCN Model → Traffic Flow ----→ INTEGRATION
                                          ↓         ↓
Time Series → LSTM Model → Travel Time --→ Optimal Route
```

---

## Running the System

### **Option 1: GUI Application (Recommended)**

```bash
python src/gui.py
```

**Features:**
- Select origin/destination on map
- Upload incident image
- View predicted severity
- See optimal route with adjusted travel time
- Compare routes with/without incidents

### **Option 2: Command Line**

```python
from src.models.cnn_model import load_cnn_model
from PIL import Image
import torch

# Load model
model = load_cnn_model('models/cnn_model.pth')

# Predict
image = Image.open('incident.jpg')
severity = model.predict(image)

print(f"Severity: {severity}")  # none/minor/moderate/severe
```

---

## Project Structure (Simplified)

```
Intro-To-AI---Assignment-2B/
├── data/
│   ├── accident_images/        # Organized dataset (train/val/test)
│   └── raw/                    # Backup of original images
│
├── models/                     # Saved model checkpoints
│   ├── cnn_model.pth          # Trained CNN (~45 MB)
│   ├── rnn_pattern_model.pth  # Trained RNN (~5 MB)
│   ├── lstm_travel_time_model.pth  # Trained LSTM (~5 MB)
│   └── gcn_model.pth          # Trained GCN (~3 MB)
│
├── src/
│   ├── train_cnn.py           # Train CNN model
│   ├── train_lstm.py          # Train LSTM/RNN models
│   ├── train_gcn.py           # Train GCN model
│   ├── data_processing.py     # Dataset utilities
│   ├── integration.py         # Combine all models
│   └── gui.py                 # Main GUI application
│
├── kaggleDataset.py           # Dataset download & organization
├── requirements.txt           # Python packages
├── README.md                  # This file
└── test_installation.py       # Verify setup
```

---

## Dataset Details

### **Sources**

1. **Clean Cars** (None class) - 2,616 images
   - Source: `kshitij192/cars-image-dataset`
   
2. **Minor Damage** - 534 + 1,000 images
   - Sources: `prajwalbhamere/...` + `abdulrahmankerim/...`
   
3. **Moderate Damage** - 538 + 1,000 images
   - Sources: `prajwalbhamere/...` + `marslanarshad/...`
   
4. **Severe Damage** - 559 + 1,500 images
   - Sources: `prajwalbhamere/...` + `exameese/...`

**Total: 6,000-8,000 images** (balanced across 4 classes)

### **After Organization**

```
data/accident_images/
├── train/ (70%)    ~4,200-5,600 images
├── val/ (15%)      ~900-1,200 images
└── test/ (15%)     ~900-1,200 images
```

---

## Model Performance

### **CNN Model (Image Classification)**
- **Architecture**: ResNet-18 (pre-trained)
- **Classes**: 4 (None, Minor, Moderate, Severe)
- **Expected Accuracy**: 90-95%
- **Training Time**: 20-30 minutes (GPU)

### **LSTM Model (Time Series)**
- **Architecture**: Bidirectional LSTM
- **Task**: Predict travel time adjustments
- **Expected MAE**: 8-12 minutes
- **Training Time**: 5-10 minutes (GPU)

### **GCN Model (Road Network)**
- **Architecture**: 2-layer Graph Convolutional Network
- **Task**: Predict traffic flow (Low/Medium/High)
- **Expected Accuracy**: 85-90%
- **Training Time**: 10-15 minutes (GPU)

---

## Common Issues & Solutions

### **Issue 1: "CUDA out of memory"**
```python
# In src/train_cnn.py, reduce batch size:
BATCH_SIZE = 16  # or 8
```

### **Issue 2: Dataset download fails**
```bash
# Check internet connection
# Or download manually from Kaggle
```

### **Issue 3: PyTorch not found**
```bash
# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### **Issue 4: Module not found errors**
```bash
# Make sure you're in project root directory
cd Intro-To-AI---Assignment-2B

# Activate virtual environment
venv\Scripts\activate  # Windows
```

### **Issue 5: Training is very slow**
- **GPU users**: Check GPU is detected with `nvidia-smi`
- **CPU users**: Consider using Google Colab (see `COLAB_GUIDE.md`)
- **Quick training**: Reduce epochs in training scripts

---

## For Teammates

### **Quick Commands**

```bash
# Setup (once)
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Get dataset (once)
python kaggleDataset.py

# Train models (in order)
python src/train_cnn.py
python src/train_lstm.py --model both
python src/train_gcn.py

# Run system
python src/gui.py
```

### **Testing Your Part**

```bash
# Test CNN (Lawrence)
python src/data_processing.py
python src/train_cnn.py

# Test LSTM (Faridz)
python src/train_lstm.py --model lstm
python demo_rnn_lstm.py

# Test GCN (Cherrylynn)
python src/train_gcn.py

# Test GUI (Jason)
python src/gui.py
```

---

## References

1. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
3. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.

**Datasets:**
- Car Damage: [Kaggle](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)
- Clean Cars: [Kaggle](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset)

---

## 📝 License

Educational project for COS30019 at Swinburne University of Technology.

---

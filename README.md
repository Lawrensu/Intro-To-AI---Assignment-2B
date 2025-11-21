# Intro To AI - Assignment 2B

## Incident Classification System (ICS)

**A comprehensive system for predicting traffic incident severity and finding optimal routes in Kuching, Malaysia.**

### Team Members
- Lawrence Lian anak Matius Ding (Team Leader)
- Mohd Faridz Faisal bin Mohd Faizal
- Cherrylynn
- Jason Hernando Kwee

---

## Quick Start

**For first-time setup, run the automated setup script:**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Intro-To-AI---Assignment-2B

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Run setup script (installs dependencies, downloads dataset)
python setup.py

# 4. Continue with Step 3 in [Installation](#installation)
```

**That's it!** The setup script handles everything automatically.

For detailed instructions, see the [Installation](#installation) and [Dataset Setup](#dataset-setup) sections below.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Model Architecture](#model-architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Dataset Setup](#dataset-setup)
7. [Training Models](#training-models)
8. [Running the System](#running-the-system)
9. [Evaluation](#evaluation)
10. [Troubleshooting](#troubleshooting)
11. [Development Timeline](#development-timeline)
12. [Team Contributions](#team-contributions)
13. [References](#references)

---

## Project Overview

This project implements machine learning models to predict traffic incident severity and integrates them with a pathfinding algorithm to provide optimal route recommendations. The system uses three different ML approaches:

1. **CNN (ResNet-18)**: For image-based incident severity detection
2. **LSTM**: For time-series traffic pattern prediction
3. **GCN**: For spatial traffic flow analysis on road networks

### Important: 3-Class Classification System

The system predicts **3 classes of traffic incident severity**: **[Minor, Moderate, Severe]**

**Note:** There is NO "none" or "no damage" class. The system only classifies existing traffic incidents, not whether an incident has occurred. This is because:
- The assignment focuses on incident severity classification
- All images in the dataset represent damaged vehicles (incidents that have occurred)
- The pathfinding system uses these severity predictions to adjust travel times

### Travel Time Adjustment Formula

```
adjusted_time = base_time × ACCIDENT_SEVERITY_MULTIPLIER
```

Where severity multipliers are:
- **Minor**: 1.2x (20% increase)
- **Moderate**: 1.5x (50% increase)
- **Severe**: 2.0x (100% increase)

### Key Components
- **Data Processing**: Handles dataset preparation, augmentation, and loading
- **Model Training**: Trains CNN, LSTM, and GCN models
- **Pathfinding Integration**: Combines ML predictions with route finding
- **GUI**: User interface for interaction and visualization

---

## Features

- Traffic incident severity prediction using 3 distinct ML models
- Integration with pathfinding algorithms from Assignment 2A
- Interactive GUI for route planning and visualization
- Model performance comparison and evaluation
- Route recommendation with incident-adjusted travel times
- Support for top-k path finding between origin and destination
- Real-time visualization using Folium maps

---

## Model Architecture

### CNN (ResNet-18)
- Pre-trained ResNet-18 with modified final layer
- Fine-tuned for traffic incident severity classification
- **Input**: RGB images of damaged vehicles (224×224)
- **Output**: 3-class probability distribution [Minor, Moderate, Severe]
- **Classes**:
  - Class 0: Minor (minor damage)
  - Class 1: Moderate (moderate damage)
  - Class 2: Severe (severe damage)
- **Training**: Transfer learning with data augmentation
- **Loss Function**: Cross-Entropy with class weights for imbalanced data

### LSTM
- Bidirectional LSTM with 2 layers
- Hidden size: 64
- **Input**: Time-series traffic data (sequence length: 30)
- **Output**: Predicted travel time adjustments
- **Use Case**: Temporal pattern recognition in traffic flow

### GCN (Graph Convolutional Network)
- 2-layer Graph Convolutional Network
- Hidden dimension: 64
- **Input**: Road network graph with node features
- **Output**: Traffic flow predictions for each road segment
- **Use Case**: Spatial relationship modeling in road networks

---

## Project Structure

```
root/
├── data/
│   ├── heritage_assignment_15_time_asymmetric-1.txt  # Road network data
│   ├── heritage_map_roads.html                        # Map visualization
│   ├── map.osm                                        # OpenStreetMap data
│   ├── raw/                                           # Raw downloaded dataset
│   │   ├── minor_damage/                             # Minor severity images
│   │   ├── moderate_damage/                          # Moderate severity images
│   │   └── severe_damage/                            # Severe severity images
│   └── accident_images/                               # Organized dataset (3 classes)
│       ├── train/
│       │   ├── minor/                                # Minor damage - training
│       │   ├── moderate/                             # Moderate damage - training
│       │   └── severe/                               # Severe damage - training
│       ├── val/
│       │   ├── minor/
│       │   ├── moderate/
│       │   └── severe/
│       └── test/
│           ├── minor/
│           ├── moderate/
│           └── severe/
├── src/
│   ├── data_processing.py                             # Dataset handling (3 classes)
│   ├── train_cnn.py                                   # CNN training script
│   ├── train_lstm.py                                  # LSTM training script 
│   ├── train_gcn.py                                   # GCN training script 
│   ├── models/
│   │   ├── cnn_model.py                              # CNN implementation (3 classes)
│   │   ├── lstm_model.py                             # LSTM implementation 
│   │   └── gcn_model.py                              # GCN implementation 
│   ├── pathfinding.py                                # Pathfinding algorithms 
│   ├── integration.py                                # ML-Pathfinding integration 
│   └── gui.py                                        # GUI implementation 
├── models/                                            # Saved model checkpoints
│   ├── cnn_model.pth                                 # Best CNN model
│   ├── cnn_training_history.png                     # Training curves
│   ├── cnn_confusion_matrix.png                     # Confusion matrix (3x3)
│   ├── lstm_model.pth
│   └── gcn_model.pth
├── tests/                                             # Unit tests
├── requirements.txt                                   # Python dependencies
├── test_installation.py                              # Installation verification
├── kaggleDataset.py                                  # Dataset download script
├── visualize_assignment_folium_roads_knearest.py     # Visualization script
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12
- **pip**: Latest version
- **NVIDIA GPU** (optional but recommended for faster training)
  - CUDA 12.1 or 12.4
  - cuDNN compatible version
- **Git**: For cloning the repository

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/traffic-incident-system.git
cd traffic-incident-system
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt indicating the virtual environment is active.

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install PyTorch

Choose the appropriate command based on your system:

**For NVIDIA GPU with CUDA 12.4:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**For NVIDIA GPU with CUDA 12.1:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (no GPU):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install PyTorch Geometric

**For CUDA 12.4:**
```bash
pip install torch-geometric==2.6.1
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

**For CUDA 12.1:**
```bash
pip install torch-geometric==2.6.1
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

**For CPU:**
```bash
pip install torch-geometric==2.6.1
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

### Step 6: Install Other Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- folium (map visualization)
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (machine learning utilities)
- matplotlib (plotting)
- seaborn (statistical visualization)
- Pillow (image processing)
- opencv-python (computer vision)
- requests (HTTP library)
- tqdm (progress bars)
- kaggle (dataset download)
- kagglehub (easier dataset download)
- jupyter (notebook interface)

### Step 7: Verify Installation

Run the installation test script:

```bash
python test_installation.py
```

Expected output should show all packages with checkmarks. If any package fails, refer to the [Troubleshooting](#troubleshooting) section.

---

## Dataset Setup

### Important: 3-Class Dataset
### Option 1: Automatic Download using KaggleHub (Easiest - Recommended)

This method doesn't require Kaggle API setup!

```bash
pip install kagglehub (should be instaled already when pip install -m requirements.txt)
python kaggleDataset.py
```

The script will:
- Download the dataset automatically (no authentication needed)
- Skip the "no damage" class
- Copy the 3 severity classes to `data/raw/`
- Organize into train/val/test splits (70%/15%/15%)
- Create the directory structure in `data/accident_images/`

### Option 2: Automatic Download using Kaggle API

1. **Set up Kaggle API credentials:**
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Download `kaggle.json`
   - Place it in:
     - **Windows**: `C:\Users\YourUsername\.kaggle\kaggle.json`
     - **Mac/Linux**: `~/.kaggle/kaggle.json`
   - Set permissions (Mac/Linux only): `chmod 600 ~/.kaggle/kaggle.json`

2. **Accept dataset terms:**
   - Visit: https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset
   - Click "Download" (you must accept terms of use)

3. **Run the dataset script:**
   ```bash
   python kaggleDataset.py
   ```

### Option 3: Manual Download

1. Visit: https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset
2. Click "Download" button
3. Extract the zip file
4. Copy only these 3 folders to `data/raw/`:
   - `01-minor` → rename to `minor_damage`
   - `02-moderate` → rename to `moderate_damage`
   - `03-severe` → rename to `severe_damage`
5. **Delete or ignore** the `00-damage` folder
6. Run the organization script:
   ```bash
   python kaggleDataset.py
   ```

### Dataset Structure After Setup

```
data/
├── raw/                          # Backup (original names)
│   ├── minor_damage/            # All minor severity images
│   ├── moderate_damage/         # All moderate severity images
│   └── severe_damage/           # All severe severity images
│
└── accident_images/             # Ready for training
    ├── train/ (70%)
    │   ├── minor/               # Minor severity - training
    │   ├── moderate/            # Moderate severity - training
    │   └── severe/              # Severe severity - training
    ├── val/ (15%)
    │   ├── minor/
    │   ├── moderate/
    │   └── severe/
    └── test/ (15%)
        ├── minor/
        ├── moderate/
        └── severe/
```

**Expected Dataset Statistics:**
- Total images: ~3,000-4,000
- Train: ~2,100-2,800 images
- Val: ~450-600 images
- Test: ~450-600 images

---

## Training Models

### 1. Train CNN Model

Train the ResNet-18 based CNN for image classification:

```bash
python src/train_cnn.py
```

**Training configuration:**
- Architecture: ResNet-18 (pre-trained on ImageNet)
- Number of classes: **3** (Minor, Moderate, Severe)
- Batch size: 32
- Learning rate: 0.001
- Number of epochs: 25
- Optimizer: Adam with weight decay (1e-4)
- Scheduler: ReduceLROnPlateau
- Early stopping: Patience of 5 epochs

**Expected output:**
```
Training CNN model on cuda...
Number of classes: 3
Class names: ['minor', 'moderate', 'severe']
Training samples: 2800
Validation samples: 600

Epoch 1/25 (45.23s)
  Train Loss: 1.2345 | Train Acc: 0.5234
  Val Loss: 1.0123 | Val Acc: 0.6012
  *** New best model! (Val Acc: 0.6012) ***

...

Training complete in 18.75 minutes
Best validation accuracy: 0.8734
Model saved to models/cnn_model.pth
```

**Generated files:**
- `models/cnn_model.pth` - Best model checkpoint
- `models/cnn_training_history.png` - Loss and accuracy curves
- `models/cnn_confusion_matrix.png` - Confusion matrix on test set (3x3)

### 2. Train LSTM Model (Coming in Day 2)

```bash
python src/train_lstm.py
```

### 3. Train GCN Model (Coming in Day 2)

```bash
python src/train_gcn.py
```

---

## Running the System

### Command Line Interface

After training models, you can use them for predictions:

```python
from src.models.cnn_model import load_cnn_model
import torch
from torchvision import transforms
from PIL import Image

# Load trained model
model = load_cnn_model('models/cnn_model.pth', device='cuda')

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/incident_image.jpg')
image_tensor = transform(image).unsqueeze(0).cuda()

# Predict
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    
classes = ['minor', 'moderate', 'severe']
print(f"Predicted Severity: {classes[prediction.item()]}")
print(f"Confidence: {probabilities[0][prediction].item():.2%}")
```

### GUI Application

```bash
python src/gui.py
```

The GUI will provide:
- Origin and destination selection
- Model selection (CNN/LSTM/GCN)
- Route visualization on map
- Incident severity predictions
- Travel time estimates with severity adjustments

---

## Evaluation

### Model Performance Metrics

**CNN Model (3-class classification):**
- Overall Accuracy
- Precision (per class)
- Recall (per class)
- F1-Score (per class)
- Confusion Matrix (3x3)
- Per-class performance:
  - Minor vs Moderate vs Severe

**LSTM Model:**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score

**GCN Model:**
- Accuracy
- Precision (per class)
- Recall (per class)
- F1-Score (per class)

### Running Evaluation

After training, evaluation is automatically performed on the test set. To manually evaluate:

```python
from src.models.cnn_model import CNNTrainer, load_cnn_model
from src.data_processing import create_dataloaders

# Load model and data
model = load_cnn_model('models/cnn_model.pth')
dataloaders, datasets = create_dataloaders('data/accident_images')

# Create trainer and evaluate
trainer = CNNTrainer(model)
results = trainer.evaluate(dataloaders['test'])

print(f"Test Accuracy: {results['accuracy']:.4f}")
print("\nPer-class metrics:")
for class_name, metrics in results['per_class_metrics'].items():
    print(f"\n{class_name.capitalize()}:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. PyTorch Installation Fails

**Problem:** CUDA version mismatch or installation errors.

**Solution:**
```bash
# Check CUDA version
nvidia-smi

# Use CPU version if GPU issues persist
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. PyTorch Geometric Installation Fails

**Problem:** Compilation errors or missing dependencies.

**Solution:**
```bash
# Try installing without pre-built wheels
pip install torch-geometric

# Or use conda (alternative)
conda install pyg -c pyg
```

#### 3. Dataset Download Issues

**Problem:** KaggleHub or Kaggle API not working.

**Solution:**
```bash
# Option 1: Try kagglehub (easier)
pip install kagglehub
python kaggleDataset.py

# Option 2: Manual download
# Follow Option 3 in Dataset Setup section
```

#### 4. "Only 0 folders found" Error

**Problem:** Script can't find image folders in downloaded dataset.

**Solution:**
This happens when the dataset has a nested structure. The updated `kaggleDataset.py` script handles this automatically. Make sure you're using the latest version:
```bash
python kaggleDataset.py
```

The script will recursively search for image folders and skip the "no damage" class.

#### 5. Dataset Has 4 Classes Instead of 3

**Problem:** Old version of data processing script.

**Solution:**
Make sure you're using the updated scripts that skip the "none/no_damage" class:
- `kaggleDataset.py` - skips 00-damage folder
- `src/data_processing.py` - uses 3 classes only
- `src/models/cnn_model.py` - NUM_CLASSES = 3

#### 6. Out of Memory (OOM) Errors

**Problem:** CUDA out of memory during training.

**Solution:**
```python
# Reduce batch size in train_cnn.py
BATCH_SIZE = 16  # or 8 for smaller GPUs

# Or use CPU
DEVICE = 'cpu'
```

#### 7. Tkinter Not Found

**Problem:** `ModuleNotFoundError: No module named 'tkinter'`

**Solution:**
- **Windows**: Reinstall Python with "tcl/tk and IDLE" option checked
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **Mac**: `brew install python-tk`

#### 8. Import Errors in Scripts

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure you're in the project root directory
cd path/to/Intro-To-AI---Assignment-2B

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Run scripts from project root
python src/train_cnn.py
```

---

## Team Contributions

### Lawrence Lian anak Matius Ding (Team Leader)
- Project management and coordination
- CNN model implementation (3-class)
- Data processing pipeline
- Integration with pathfinding

### Mohd Faridz Faisal bin Mohd Faizal
- LSTM model implementation
- Time-series data generation
- Model evaluation and testing

### Cherrylynn 
- GCN model implementation
- Graph construction from road network
- Spatial analysis

### Jason Hernando Kwee
- GUI implementation
- Visualization with Folium
- User interface design

---

## Acknowledgements

This project is part of the **COS30019 Introduction to AI** course at **Swinburne University of Technology, Sarawak Campus**.

Special thanks to:
- Dr. Joel for guidance and support
- Kaggle community for the car damage dataset
- OpenStreetMap contributors for map data

---

## References

### Academic Papers
1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition**. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.

2. Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*, 9(8), 1735-1780.

3. Kipf, T. N., & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks**. *International Conference on Learning Representations (ICLR)*.

### Datasets
4. Bhamere, P. (2023). **Car Damage Severity Dataset**. Kaggle. https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset

### Libraries and Frameworks
5. Paszke, A., et al. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library**. *Advances in Neural Information Processing Systems*, 32.

6. Fey, M., & Lenssen, J. E. (2019). **Fast Graph Representation Learning with PyTorch Geometric**. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.

7. Folium Contributors. (2024). **Folium: Python Data, Leaflet.js Maps**. https://python-visualization.github.io/folium/

---

## License

This project is developed for educational purposes as part of COS30019 coursework at Swinburne University of Technology.

---

Last Updated: December 2024
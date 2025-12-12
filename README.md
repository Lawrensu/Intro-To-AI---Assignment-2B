# Traffic Incident Classification System

**COS30019 - Introduction to AI - Assignment 2B**

An AI-powered system for traffic incident severity classification and intelligent route optimization for Kuching Heritage Area, Malaysia.

---

## System Overview

### Purpose
This system addresses real-world traffic management challenges by:
1. Classifying accident severity from images using deep learning
2. Adjusting travel time predictions based on incident severity
3. Providing optimal route recommendations using multiple pathfinding algorithms

### Core Functionality
- **Image Analysis**: Ensemble of 3 CNN models (ResNet-18, MobileNet-V2, EfficientNet-B0)
- **Severity Classification**: 4-class system (none, minor, moderate, severe)
- **Dynamic Route Planning**: 6 pathfinding algorithms with severity-based edge weight adjustment
- **Interactive Visualization**: Dark-themed GUI with landmark-based navigation

---

## Team Members

- Lawrence Lian anak Matius Ding (Team Leader)
- Mohd Faridz Faisal
- Cherrylyn Munai 
- Jason Hernando

---

## Technical Architecture

### Machine Learning Models

**1. Image Classification (3 CNNs)**
- **ResNet-18**: Baseline architecture with residual connections
- **MobileNet-V2**: Lightweight model for efficient inference
- **EfficientNet-B0**: State-of-the-art compound scaling architecture
- **Reasoning**: Ensemble approach improves prediction reliability through model diversity

**2. Severity Impact System**

| Severity | Multiplier | Impact | Use Case |
|----------|-----------|--------|----------|
| None | 1.0x | No delay | Normal traffic conditions |
| Minor | 1.2x | +20% time | Fender bender, minor collision |
| Moderate | 1.5x | +50% time | Lane blockage, moderate damage |
| Severe | 2.0x | +100% time | Multi-vehicle accident, road closure |

**Reasoning**: Multipliers based on real-world traffic impact studies where severe incidents can double travel time due to lane closures and congestion effects.

**3. Pathfinding Integration (6 Algorithms)**
- BFS, DFS, UCS, GBFS, A*, IDA*
- **Reasoning**: Multiple algorithms allow comparison of optimality vs computational efficiency trade-offs

---

## Installation & Setup

### System Requirements
- Python 3.10 or higher (3.11 recommended)
- 8GB RAM minimum (16GB recommended for training)
- GPU with CUDA support (optional, significantly faster for training)
- 5GB disk space (dataset + models)

### Environment Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd Intro-To-AI---Assignment-2B

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows PowerShell:
[Activate.ps1](http://_vscodecontentref_/1)
# Windows CMD:
[activate.bat](http://_vscodecontentref_/2)
# Linux/Mac:
source venv/bin/activate

# 4. Install PyTorch
# For NVIDIA GPU (CUDA 12.4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Install other dependencies
pip install -r [requirements.txt](http://_vscodecontentref_/3)

# 6. Verify installation
python [test_installation.py](http://_vscodecontentref_/4)

# 7. Dataset Acquisition
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
python [kaggleDataset.py](http://_vscodecontentref_/5)
```

## Model Training
Train all image models
```
python src/train_all_models.py
```

## Running the System
GUI Application
```
python src/gui.py
```

### Features:
1. Upload incident image
2. Get severity predictions from 3 moedls with ensemble voting
3. Select origin/destination using landmark names (e.g., "1: Fort Margherita")
4. View before/after travel time comparison
5. Display top-5 optimal routes
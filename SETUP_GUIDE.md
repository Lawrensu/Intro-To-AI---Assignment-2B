# Complete Setup Guide 🚀

**Last Updated**: December 2024

---

## Prerequisites

- **Python 3.10, 3.11, or 3.12** ([Download](https://www.python.org/downloads/))
- **8GB+ RAM** (16GB recommended)
- **10GB free disk space** (for dataset)
- **NVIDIA GPU** (optional but recommended - speeds up training 10x)
- **Internet connection** (for dataset download)

---

## Step-by-Step Setup

### **Step 1: Install Python**

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```bash
   python --version
   # Should show: Python 3.10.x or 3.11.x or 3.12.x
   ```

### **Step 2: Clone Repository**

```bash
# Option A: Using Git
git clone <your-repo-url>
cd Intro-To-AI---Assignment-2B

# Option B: Download ZIP
# 1. Download ZIP from GitHub
# 2. Extract to a folder
# 3. Open terminal in that folder
```

### **Step 3: Create Virtual Environment**

**Windows (PowerShell/CMD):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Success:** You should see `(venv)` in your terminal prompt

### **Step 4: Install PyTorch**

**Choose ONE command based on your system:**

**A. NVIDIA GPU (CUDA 12.4) - FASTEST:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**B. NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**C. CPU Only (No GPU) - SLOWER:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

**How to check CUDA version:**
```bash
nvidia-smi
# Look for "CUDA Version: 12.x"
```

### **Step 5: Install PyTorch Geometric**

**A. For CUDA 12.4:**
```bash
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

**B. For CUDA 12.1:**
```bash
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

**C. For CPU:**
```bash
pip install torch-geometric==2.6.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

### **Step 6: Install Other Packages**

```bash
pip install -r requirements.txt
```

This installs:
- numpy, pandas, scikit-learn (data science)
- matplotlib, seaborn, folium (visualization)
- Pillow, opencv-python (image processing)
- kaggle, kagglehub (dataset download)

### **Step 7: Verify Installation**

```bash
python test_installation.py
```

**Expected output:**
```
✓ Python: 3.11.x
✓ PyTorch: 2.5.1
✓ CUDA: Available (or Not available for CPU)
✓ PyTorch Geometric: 2.6.1
✓ NumPy: 1.24.x
... (all packages should have ✓)
```

---

## Download Dataset

### **Step 1: Run Dataset Script**

```bash
python kaggleDataset.py
```

**What happens:**
1. Downloads 5 datasets from Kaggle (~6-8 GB)
2. Organizes into 4 classes (None, Minor, Moderate, Severe)
3. Splits into train/val/test (70%/15%/15%)
4. **Time**: 10-30 minutes depending on internet speed

### **Step 2: Verify Dataset**

```bash
python src/data_processing.py
```

**Expected output:**
```
Train set: 4200-5600 images
  none:     1050-1400 images
  minor:    1050-1400 images
  moderate: 1050-1400 images
  severe:   1050-1400 images

[OK] Dataset is sufficient!
```

---

## Train Models

### **Step 1: Train CNN (20-30 minutes)**

```bash
python src/train_cnn.py
```

**Progress:**
```
Epoch 1/25 (45.2s)
  Train Loss: 1.234 | Train Acc: 0.523
  Val Loss: 1.012 | Val Acc: 0.612
  *** New best model! ***
...
Training complete in 18.5 minutes
Best validation accuracy: 0.925
```

**Output files:**
- `models/cnn_model.pth` - Trained model
- `models/cnn_training_history.png` - Loss/accuracy curves
- `models/cnn_confusion_matrix.png` - 4×4 confusion matrix

### **Step 2: Train LSTM (5-10 minutes)**

```bash
python src/train_lstm.py --model both
```

**Output files:**
- `models/rnn_pattern_model.pth` - Traffic pattern RNN
- `models/lstm_travel_time_model.pth` - Travel time LSTM

### **Step 3: Train GCN (10-15 minutes)**

```bash
python src/train_gcn.py
```

**Output files:**
- `models/gcn_model.pth` - Traffic flow GCN
- `models/gcn_training_history.png`
- `models/gcn_confusion_matrix.png`

---

## Verify Everything Works

```bash
# Test CNN predictions
python -c "from src.models.cnn_model import load_cnn_model; print('CNN: OK')"

# Test LSTM predictions
python demo_rnn_lstm.py

# Test GUI
python src/gui.py
```

---

## Troubleshooting

### **Problem: "Python not found"**
**Solution:** Add Python to PATH
1. Windows: Reinstall Python, check "Add to PATH"
2. Mac: `export PATH="/usr/local/bin/python3:$PATH"`
3. Linux: `sudo apt-get install python3`

### **Problem: "pip not found"**
**Solution:**
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### **Problem: "CUDA out of memory"**
**Solution:** Reduce batch size
```python
# In src/train_cnn.py:
BATCH_SIZE = 16  # or 8
```

### **Problem: "No module named 'torch'"**
**Solution:** Reinstall PyTorch
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### **Problem: Dataset download fails**
**Solution:** Check internet connection or download manually from:
- https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset
- https://www.kaggle.com/datasets/kshitij192/cars-image-dataset

---

## Tips

1. **Always activate virtual environment** before working
2. **Use GPU** for 10x faster training (CUDA version)
3. **Monitor training** with the generated PNG plots
4. **Save checkpoints** regularly (models auto-save best)
5. **Check logs** if training fails

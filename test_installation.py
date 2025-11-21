"""
Quick script to verify all packages are installed correctly
Updated for latest versions (December 2024)
"""

import sys
from pathlib import Path
import os

print("="*70)
print("Traffic Incident Classification System - Installation Test")
print("="*70)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("="*70)
print("\nTesting package imports...\n")

packages_status = []

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print(f"  Running on CPU (training will be slower)")
    packages_status.append(("PyTorch", True))
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    packages_status.append(("PyTorch", False))

# Test TorchVision
try:
    import torchvision
    print(f"✓ TorchVision: {torchvision.__version__}")
    packages_status.append(("TorchVision", True))
except ImportError as e:
    print(f"✗ TorchVision: {e}")
    packages_status.append(("TorchVision", False))

# Test PyTorch Geometric
try:
    import torch_geometric
    print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
    
    # Test if we can import key components
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    print(f"  - GCNConv: Available")
    print(f"  - Data: Available")
    packages_status.append(("PyTorch Geometric", True))
except ImportError as e:
    print(f"✗ PyTorch Geometric: {e}")
    packages_status.append(("PyTorch Geometric", False))

# Test PyG extensions
try:
    import torch_scatter
    import torch_sparse
    import torch_cluster
    print(f"✓ PyTorch Geometric Extensions:")
    print(f"  - torch_scatter: {torch_scatter.__version__}")
    print(f"  - torch_sparse: {torch_sparse.__version__}")
    print(f"  - torch_cluster: {torch_cluster.__version__}")
    packages_status.append(("PyG Extensions", True))
except ImportError as e:
    print(f"✗ PyTorch Geometric Extensions: {e}")
    packages_status.append(("PyG Extensions", False))

# Test NumPy
try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
    packages_status.append(("NumPy", True))
except ImportError as e:
    print(f"✗ NumPy: {e}")
    packages_status.append(("NumPy", False))

# Test Pandas
try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
    packages_status.append(("Pandas", True))
except ImportError as e:
    print(f"✗ Pandas: {e}")
    packages_status.append(("Pandas", False))

# Test Matplotlib
try:
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
    packages_status.append(("Matplotlib", True))
except ImportError as e:
    print(f"✗ Matplotlib: {e}")
    packages_status.append(("Matplotlib", False))

# Test Seaborn
try:
    import seaborn as sns
    print(f"✓ Seaborn: {sns.__version__}")
    packages_status.append(("Seaborn", True))
except ImportError as e:
    print(f"✗ Seaborn: {e}")
    packages_status.append(("Seaborn", False))

# Test Pillow
try:
    import PIL
    print(f"✓ Pillow: {PIL.__version__}")
    packages_status.append(("Pillow", True))
except ImportError as e:
    print(f"✗ Pillow: {e}")
    packages_status.append(("Pillow", False))

# Test OpenCV
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
    packages_status.append(("OpenCV", True))
except ImportError as e:
    print(f"✗ OpenCV: {e}")
    packages_status.append(("OpenCV", False))

# Test Scikit-learn
try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
    packages_status.append(("Scikit-learn", True))
except ImportError as e:
    print(f"✗ Scikit-learn: {e}")
    packages_status.append(("Scikit-learn", False))

# Test Folium
try:
    import folium
    print(f"✓ Folium: {folium.__version__}")
    packages_status.append(("Folium", True))
except ImportError as e:
    print(f"✗ Folium: {e}")
    packages_status.append(("Folium", False))

# Test Tkinter
try:
    import tkinter as tk
    print(f"✓ Tkinter: Available (version: {tk.TkVersion})")
    packages_status.append(("Tkinter", True))
except ImportError as e:
    print(f"✗ Tkinter: {e}")
    print("  Install: sudo apt-get install python3-tk (Linux) or reinstall Python (Windows)")
    packages_status.append(("Tkinter", False))

# Test Requests
try:
    import requests
    print(f"✓ Requests: {requests.__version__}")
    packages_status.append(("Requests", True))
except ImportError as e:
    print(f"✗ Requests: {e}")
    packages_status.append(("Requests", False))

# Test TQDM
try:
    import tqdm
    print(f"✓ TQDM: {tqdm.__version__}")
    packages_status.append(("TQDM", True))
except ImportError as e:
    print(f"✗ TQDM: {e}")
    packages_status.append(("TQDM", False))

# Test Kaggle (without authentication)
try:
    # Check if kaggle package is installed without triggering authentication
    import importlib.util
    kaggle_spec = importlib.util.find_spec("kaggle")
    if kaggle_spec is not None:
        print(f"✓ Kaggle API: Installed")
        
        # Check if kaggle.json exists
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if kaggle_json.exists():
            print(f"  - Credentials: Found at {kaggle_json}")
        else:
            print(f"  - Credentials: NOT FOUND at {kaggle_json}")
            print(f"  - You'll need to set up kaggle.json before downloading datasets")
        
        packages_status.append(("Kaggle", True))
    else:
        print(f"✗ Kaggle API: Not installed")
        packages_status.append(("Kaggle", False))
except Exception as e:
    print(f"✗ Kaggle API: {e}")
    packages_status.append(("Kaggle", False))

# Check directory structure
print("\n" + "="*70)
print("Checking Directory Structure...")
print("="*70)

required_dirs = [
    "data",
    "data/accident_images",
    "src",
    "src/models",
    "models",
    "tests"
]

for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"✓ {dir_path}/")
    else:
        print(f"✗ {dir_path}/ (missing)")
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  → Created {dir_path}/")

# Summary
print("\n" + "="*70)
print("INSTALLATION SUMMARY")
print("="*70)

success_count = sum(1 for _, status in packages_status if status)
total_count = len(packages_status)

print(f"Successfully installed: {success_count}/{total_count} packages")

if success_count == total_count:
    print("\n✓ All packages installed successfully!")
    print("\n✓ You're ready to proceed!")
    print("\nNext steps:")
    
    # Check if kaggle.json exists
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("\n[IMPORTANT] Kaggle API Setup Required:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print(f"4. Place it at: {kaggle_json}")
        print("5. Then run: python kaggleDataset.py")
    else:
        print("1. Download dataset: python kaggleDataset.py")
    
    print("2. Train CNN model: python src/train_cnn.py")
else:
    print("\n⚠ Some packages failed to install:")
    for package, status in packages_status:
        if not status:
            print(f"  ✗ {package}")
    print("\nPlease check the error messages above and reinstall missing packages.")

print("="*70)

# Quick GPU test if available
try:
    import torch
    if torch.cuda.is_available():
        print("\nGPU Performance Test:")
        print("-" * 70)
        try:
            import time
            
            # Warm-up
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            _ = torch.matmul(x, y)
            torch.cuda.synchronize()
            
            # Actual test
            x = torch.randn(5000, 5000).cuda()
            y = torch.randn(5000, 5000).cuda()
            
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"✓ GPU matrix multiplication (5000x5000): {elapsed:.4f}s")
            
            # Memory info
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"✓ GPU memory: {memory_allocated:.2f} MB allocated, {memory_reserved:.2f} MB reserved")
            print("✓ GPU is working properly!")
        except Exception as e:
            print(f"✗ GPU test failed: {e}")
        print("="*70)
except:
    pass
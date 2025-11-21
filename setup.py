"""
Setup script for Intro To AI - Assignment 2B
Traffic Incident Classification System

This script:
1. Creates necessary directory structure
2. Downloads the dataset from Kaggle
3. Organizes data into train/val/test splits
4. Verifies installation

Usage:
    python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def print_header(message):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)


def print_success(message):
    """Print success message"""
    print(f"✓ {message}")


def print_error(message):
    """Print error message"""
    print(f"✗ {message}")


def print_warning(message):
    """Print warning message"""
    print(f"⚠ {message}")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print_error(f"Python 3.10+ required. Current: {version.major}.{version.minor}")
        print("Please install Python 3.10, 3.11, or 3.12")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_virtual_environment():
    """Check if virtual environment is activated"""
    print_header("Checking Virtual Environment")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Virtual environment is active")
        return True
    else:
        print_warning("Virtual environment is NOT active")
        print("\nRecommendation: Activate virtual environment first:")
        print("  Windows: venv\\Scripts\\activate")
        print("  Mac/Linux: source venv/bin/activate")
        
        response = input("\nContinue anyway? (y/n): ")
        return response.lower() == 'y'


def create_directory_structure():
    """Create necessary directories with .gitkeep files"""
    print_header("Creating Directory Structure")
    
    directories = [
        "data",
        "data/raw",
        "data/accident_images",
        "data/accident_images/train",
        "data/accident_images/val",
        "data/accident_images/test",
        "models",
        "src",
        "src/models",
        "tests",
        "logs",
        "visualizations"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep to preserve directory structure in Git
        gitkeep = path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
        
        print_success(f"Created: {directory}/")
    
    return True


def check_requirements():
    """Check if requirements.txt exists"""
    print_header("Checking Requirements File")
    
    if not Path("requirements.txt").exists():
        print_error("requirements.txt not found!")
        print("Please ensure requirements.txt is in the project root")
        return False
    
    print_success("requirements.txt found")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    print("This will install packages from requirements.txt...")
    print("This may take several minutes.\n")
    
    response = input("Proceed with installation? (y/n): ")
    if response.lower() != 'y':
        print_warning("Skipping dependency installation")
        return False
    
    try:
        # Upgrade pip first
        print("\nUpgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        print_success("pip upgraded")
        
        # Install requirements
        print("\nInstalling requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        print_success("All dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        print("\nTry installing manually:")
        print("  pip install -r requirements.txt")
        return False


def verify_installation():
    """Verify that all required packages are installed"""
    print_header("Verifying Installation")
    
    if Path("test_installation.py").exists():
        try:
            subprocess.run([sys.executable, "test_installation.py"], check=True)
            return True
        except subprocess.CalledProcessError:
            print_warning("Some packages may not be installed correctly")
            return False
    else:
        print_warning("test_installation.py not found, skipping verification")
        return True


def setup_dataset():
    """Download and organize the dataset"""
    print_header("Dataset Setup")
    
    # Check if dataset already exists
    accident_images = Path("data/accident_images")
    if (accident_images / "train").exists() and len(list((accident_images / "train").rglob("*.jpg"))) > 0:
        print_success("Dataset already organized")
        
        response = input("\nRe-download and organize dataset? (y/n): ")
        if response.lower() != 'y':
            return True
    
    print("\nThis will download the Car Damage Severity Dataset from Kaggle.")
    print("Dataset size: ~500MB")
    print("Note: Only 3 severity classes will be used (Minor, Moderate, Severe)")
    
    response = input("\nProceed with dataset download? (y/n): ")
    if response.lower() != 'y':
        print_warning("Skipping dataset setup")
        print("\nYou can run dataset setup later with:")
        print("  python kaggleDataset.py")
        return False
    
    if not Path("kaggleDataset.py").exists():
        print_error("kaggleDataset.py not found!")
        return False
    
    try:
        subprocess.run([sys.executable, "kaggleDataset.py"], check=True)
        print_success("Dataset downloaded and organized")
        return True
    except subprocess.CalledProcessError:
        print_error("Dataset setup failed")
        print("\nTry running manually:")
        print("  python kaggleDataset.py")
        return False


def print_next_steps():
    """Print next steps after setup"""
    print_header("Setup Complete!")
    
    print("\n✓ Directory structure created")
    print("✓ Dependencies installed (if selected)")
    print("✓ Dataset organized (if selected)")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    print("\n1. Verify your installation:")
    print("   python test_installation.py")
    
    print("\n2. Test data processing:")
    print("   python src/data_processing.py")
    
    print("\n3. Train the CNN model:")
    print("   python src/train_cnn.py")
    
    print("\n4. Check the README for more information:")
    print("   Open README.md in your editor")
    
    print("\n" + "=" * 70)
    print("IMPORTANT NOTES")
    print("=" * 70)
    print("• The system uses 3 classes: Minor, Moderate, Severe")
    print("• No 'none' class - we only classify existing incidents")
    print("• Training requires ~4GB GPU memory (or use CPU)")
    print("• Expected training time: 15-25 minutes on GPU")
    print("=" * 70 + "\n")


def main():
    """Main setup function"""
    print_header("Intro To AI - Assignment 2B Setup")
    print("Intelligent Traffic Incident Classification System")
    print("\nThis script will set up your environment for the project.")
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Check virtual environment
    if not check_virtual_environment():
        return
    
    # Step 3: Create directory structure
    if not create_directory_structure():
        print_error("Failed to create directory structure")
        return
    
    # Step 4: Check requirements.txt
    if not check_requirements():
        return
    
    # Step 5: Install dependencies
    install_deps = input("\nInstall Python dependencies? (y/n): ")
    if install_deps.lower() == 'y':
        install_dependencies()
        verify_installation()
    
    # Step 6: Setup dataset
    setup_dataset()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
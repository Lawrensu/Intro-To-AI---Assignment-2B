"""
Script to download and prepare the Kaggle Car Damage Severity Dataset
Dataset: https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset

This dataset is used as a proxy for traffic incident severity classification.
The car damage levels map to traffic incident severity levels.

This script:
1. Downloads dataset from Kaggle (to kagglehub cache)
2. Copies to data/raw/ (backup, all images in class folders)
3. Organizes into data/accident_images/ (train/val/test splits for training)

CLASS STRUCTURE (3 classes):
- data/raw/: minor_damage, moderate_damage, severe_damage
- data/accident_images/: minor, moderate, severe

Note: There is NO "none" class - we only classify existing incidents.
"""

import os
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing import prepare_kaggle_dataset


def explore_downloaded_structure(download_path, depth=0, max_depth=3):
    """
    Explore the structure of the downloaded dataset recursively
    
    Args:
        download_path (str): Path where dataset was downloaded
        depth (int): Current depth level
        max_depth (int): Maximum depth to explore
    
    Returns:
        dict: Structure information
    """
    path = Path(download_path)
    structure = {
        'folders': [],
        'files': [],
        'total_images': 0,
        'nested_structure': {}
    }
    
    if not path.exists() or depth >= max_depth:
        return structure
    
    indent = "  " * depth
    
    if depth == 0:
        print(f"\nExploring downloaded dataset structure at: {path}")
        print("-" * 70)
    
    # List all items in the download path
    for item in path.iterdir():
        if item.is_dir():
            structure['folders'].append(item.name)
            img_count = len(list(item.glob('*.[jJ][pP][gG]'))) + \
                       len(list(item.glob('*.[jJ][pP][eE][gG]'))) + \
                       len(list(item.glob('*.[pP][nN][gG]')))
            
            # Check if folder has subfolders
            subfolders = [x for x in item.iterdir() if x.is_dir()]
            
            if len(subfolders) > 0:
                print(f"{indent}Folder: {item.name}/ ({len(subfolders)} subfolders)")
                # Recursively explore
                nested = explore_downloaded_structure(item, depth + 1, max_depth)
                structure['nested_structure'][item.name] = nested
                structure['total_images'] += nested['total_images']
            else:
                print(f"{indent}Folder: {item.name}/ ({img_count} images)")
                structure['total_images'] += img_count
        else:
            structure['files'].append(item.name)
            if depth == 0:
                print(f"{indent}File: {item.name}")
    
    if depth == 0:
        print("-" * 70)
        print(f"Total folders: {len(structure['folders'])}")
        print(f"Total files: {len(structure['files'])}")
        print(f"Total images: {structure['total_images']}")
    
    return structure


def find_image_folders(root_path, min_images=10):
    """
    Recursively find folders containing images
    
    Args:
        root_path (Path): Root path to search
        min_images (int): Minimum number of images to consider a valid folder
    
    Returns:
        list: List of tuples (folder_path, image_count, folder_name)
    """
    image_folders = []
    
    def search_recursive(current_path):
        for item in current_path.iterdir():
            if item.is_dir():
                # Count images in this folder
                img_count = len(list(item.glob('*.[jJ][pP][gG]'))) + \
                           len(list(item.glob('*.[jJ][pP][eE][gG]'))) + \
                           len(list(item.glob('*.[pP][nN][gG]')))
                
                if img_count >= min_images:
                    image_folders.append((item, img_count, item.name))
                
                # Continue searching in subfolders
                search_recursive(item)
    
    search_recursive(root_path)
    return image_folders


def copy_kagglehub_dataset_to_raw(kagglehub_path, raw_path="data/raw"):
    """
    Copy dataset from kagglehub cache to our data/raw directory
    Only copies the 3 incident severity classes (no "none" class)
    
    Args:
        kagglehub_path (str): Path where kagglehub downloaded the dataset
        raw_path (str): Our target raw data directory
    
    Returns:
        bool: True if successful
    """
    print(f"\nCopying dataset from kagglehub cache to {raw_path}...")
    print("(Only incident severity classes: minor, moderate, severe)")
    
    kagglehub_dir = Path(kagglehub_path)
    raw_dir = Path(raw_path)
    
    # Create raw directory
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # First, explore what we actually have
    structure = explore_downloaded_structure(kagglehub_path)
    
    # Find all folders with images
    image_folders = find_image_folders(kagglehub_dir)
    
    if len(image_folders) == 0:
        print("No folders with images found in downloaded dataset!")
        return False
    
    print(f"\nFound {len(image_folders)} folders with images:")
    for folder_path, img_count, folder_name in image_folders:
        print(f"  {folder_name}: {img_count} images")
    
    # Map Kaggle folder names to standard raw folder names
    # ONLY map the 3 severity classes (skip no_damage/00-damage)
    folder_mapping = {
        '01-minor': 'minor_damage',
        'minor_damage': 'minor_damage',
        'minor damage': 'minor_damage',
        'minor': 'minor_damage',
        '1': 'minor_damage',
        '02-moderate': 'moderate_damage',
        'moderate_damage': 'moderate_damage',
        'moderate damage': 'moderate_damage',
        'moderate': 'moderate_damage',
        '2': 'moderate_damage',
        '03-severe': 'severe_damage',
        'severe_damage': 'severe_damage',
        'severe damage': 'severe_damage',
        'severe': 'severe_damage',
        '3': 'severe_damage'
    }
    
    copied_folders = []
    skipped_folders = []
    
    print(f"\nCopying folders to {raw_path}:")
    
    # Copy all image folders, mapping to our standard names
    for folder_path, img_count, folder_name in image_folders:
        folder_lower = folder_name.lower()
        
        # Skip "no damage" folders
        if any(skip in folder_lower for skip in ['00-damage', 'no_damage', 'no damage', 'none']):
            print(f"  Skipping '{folder_name}' (not used in incident classification)")
            skipped_folders.append(folder_name)
            continue
        
        # Determine destination name
        dest_name = folder_mapping.get(folder_lower, None)
        
        if dest_name is None:
            print(f"  Skipping '{folder_name}' (unknown class)")
            skipped_folders.append(folder_name)
            continue
        
        dest = raw_dir / dest_name
        
        # Remove destination if it exists
        if dest.exists():
            print(f"  Removing existing {dest_name}...")
            shutil.rmtree(dest)
        
        # Copy folder
        print(f"  Copying '{folder_name}' -> '{dest_name}' ({img_count} images)...")
        shutil.copytree(folder_path, dest)
        copied_folders.append(dest_name)
    
    if skipped_folders:
        print(f"\nSkipped folders: {skipped_folders}")
    
    if len(copied_folders) > 0:
        print(f"\nSuccessfully copied {len(copied_folders)} incident severity classes to {raw_path}")
        print(f"Classes: {copied_folders}")
        
        # Count total images
        total_images = count_images(raw_path)
        print(f"Total images: {total_images}")
        
        return True
    else:
        print("No valid incident severity folders found to copy")
        return False


def download_with_kagglehub(dataset_name="prajwalbhamere/car-damage-severity-dataset"):
    """
    Download dataset using kagglehub (newer, easier method)
    
    Args:
        dataset_name (str): Kaggle dataset identifier
    
    Returns:
        tuple: (success: bool, download_path: str)
    """
    try:
        import kagglehub
        
        print("Using kagglehub to download dataset...")
        print(f"Dataset: {dataset_name}")
        print("This may take a few minutes depending on your internet speed...")
        print()
        
        # Download dataset - kagglehub handles everything automatically
        path = kagglehub.dataset_download(dataset_name)
        
        print(f"\nDataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        
        return True, path
        
    except ImportError:
        print("kagglehub is not installed.")
        print("Install it with: pip install kagglehub")
        return False, None
    except Exception as e:
        print(f"Error downloading with kagglehub: {e}")
        print("\nTrying traditional Kaggle API method...")
        return False, None


def download_with_kaggle_api(dataset_name, download_path="data/raw"):
    """
    Download dataset using traditional Kaggle API
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        download_path (str): Path to download the dataset
    
    Returns:
        tuple: (success: bool, download_path: str)
    """
    print(f"Using Kaggle API to download dataset...")
    print(f"Dataset: {dataset_name}")
    print(f"Download path: {download_path}")
    
    # Create download directory
    Path(download_path).mkdir(parents=True, exist_ok=True)
    
    try:
        import subprocess
        
        # Download using Kaggle API
        cmd = f"kaggle datasets download -d {dataset_name} -p {download_path} --unzip"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Download successful!")
            return True, download_path
        else:
            print(f"Download failed: {result.stderr}")
            return False, None
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease ensure:")
        print("1. Kaggle API is installed: pip install kaggle")
        print("2. Kaggle API token is configured in ~/.kaggle/kaggle.json")
        print("3. You have accepted the dataset's terms on Kaggle website")
        return False, None


def manual_download_instructions():
    """Print instructions for manual dataset download"""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("\n1. Visit: https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset")
    print("2. Click 'Download' button")
    print("3. Extract the zip file to: data/raw/")
    print("4. Keep only these 3 folders (delete no_damage/00-damage):")
    print("   - 01-minor or minor_damage")
    print("   - 02-moderate or moderate_damage")
    print("   - 03-severe or severe_damage")
    print("\n5. After extraction, run this script again")
    print("=" * 70)


def check_dataset_exists(raw_path="data/raw"):
    """
    Check if dataset already exists in raw directory
    Expects 3 classes: minor_damage, moderate_damage, severe_damage
    
    Args:
        raw_path (str): Path to raw dataset
    
    Returns:
        bool: True if dataset exists, False otherwise
    """
    raw_dir = Path(raw_path)
    
    # Check for our 3 incident severity classes
    expected_folders = ['minor_damage', 'moderate_damage', 'severe_damage']
    
    if not raw_dir.exists():
        return False
    
    # Check if all expected folders exist and have images
    all_exist = all((raw_dir / folder).exists() for folder in expected_folders)
    if all_exist:
        # Check if folders have images
        has_images = all(
            len(list((raw_dir / folder).glob('*.[jJ][pP][gG]'))) > 0 or
            len(list((raw_dir / folder).glob('*.[pP][nN][gG]'))) > 0
            for folder in expected_folders
        )
        if has_images:
            return True
    
    # Check if there are any folders with images (alternative naming)
    for item in raw_dir.iterdir():
        if item.is_dir():
            img_files = list(item.glob('*.[jJ][pP][gG]')) + \
                       list(item.glob('*.[pP][nN][gG]'))
            if len(img_files) > 10:  # At least 10 images
                return True
    
    return False


def organize_dataset(raw_path="data/raw", target_path="data/accident_images"):
    """
    Organize the downloaded dataset into train/val/test splits
    Maps from raw folder names to our training folder names:
    - minor_damage -> minor
    - moderate_damage -> moderate
    - severe_damage -> severe
    
    3 CLASSES ONLY (no "none" class)
    
    Args:
        raw_path (str): Path to raw downloaded dataset
        target_path (str): Path to organized dataset
    
    Returns:
        bool: True if successful
    """
    print("\n" + "=" * 70)
    print("Organizing dataset into train/val/test splits...")
    print("Mapping: raw folder names -> training folder names")
    print("  minor_damage -> minor")
    print("  moderate_damage -> moderate")
    print("  severe_damage -> severe")
    print("\nNOTE: 3 classes only (Minor, Moderate, Severe)")
    print("=" * 70)
    
    # Check if raw dataset exists
    if not check_dataset_exists(raw_path):
        print(f"Error: Dataset not found in {raw_path}")
        print("Expected 3 folders: minor_damage, moderate_damage, severe_damage")
        return False
    
    # Prepare dataset
    try:
        prepare_kaggle_dataset(
            kaggle_dir=raw_path,
            target_dir=target_path,
            train_split=0.7,
            val_split=0.15
        )
        print(f"\nDataset organized successfully in {target_path}")
        
        # Show final structure
        print("\nFinal structure (3 classes):")
        print(f"{target_path}/")
        for split in ['train', 'val', 'test']:
            split_path = Path(target_path) / split
            if split_path.exists():
                print(f"  {split}/")
                for class_folder in ['minor', 'moderate', 'severe']:
                    class_path = split_path / class_folder
                    if class_path.exists():
                        count = len(list(class_path.glob('*.jpg'))) + len(list(class_path.glob('*.png')))
                        print(f"    {class_folder}/ ({count} images)")
        
        return True
    except Exception as e:
        print(f"Error organizing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def count_images(directory):
    """Count total images in a directory"""
    path = Path(directory)
    if not path.exists():
        return 0
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    count = 0
    
    for ext in image_extensions:
        count += len(list(path.rglob(f'*{ext}')))
        count += len(list(path.rglob(f'*{ext.upper()}')))
    
    return count


def main():
    print("=" * 70)
    print("Kaggle Dataset Preparation Script")
    print("Traffic Incident Severity Classification Dataset")
    print("=" * 70)
    print("\nIMPORTANT: This dataset has 3 classes (not 4)")
    print("Classes: Minor, Moderate, Severe")
    print("(No 'none' class - we only classify existing incidents)")
    print("=" * 70)
    print("\nThis script performs three steps:")
    print("1. Download from Kaggle (to kagglehub cache)")
    print("2. Copy to data/raw/ (3 severity classes only)")
    print("3. Organize to data/accident_images/ (train/val/test splits)")
    print("=" * 70)
    
    RAW_PATH = "data/raw"
    TARGET_PATH = "data/accident_images"
    DATASET_NAME = "prajwalbhamere/car-damage-severity-dataset"
    
    # Check if organized dataset already exists
    if check_dataset_exists(TARGET_PATH + "/train"):
        print(f"\nOrganized dataset already exists in {TARGET_PATH}")
        train_count = count_images(TARGET_PATH + "/train")
        val_count = count_images(TARGET_PATH + "/val")
        test_count = count_images(TARGET_PATH + "/test")
        print(f"  Train images: {train_count}")
        print(f"  Val images: {val_count}")
        print(f"  Test images: {test_count}")
        print(f"  Total: {train_count + val_count + test_count}")
        
        response = input("\nDo you want to re-organize the dataset? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Check if raw dataset exists
    if check_dataset_exists(RAW_PATH):
        print(f"\nRaw dataset found in {RAW_PATH}")
        raw_count = count_images(RAW_PATH)
        print(f"Total images in raw dataset: {raw_count}")
        
        # Organize dataset
        if organize_dataset(RAW_PATH, TARGET_PATH):
            print("\n" + "=" * 70)
            print("SUCCESS! Dataset is ready for training")
            print("Dataset has 3 classes: Minor, Moderate, Severe")
            print("=" * 70)
        
    else:
        print(f"\nRaw dataset not found in {RAW_PATH}")
        print("\nAttempting to download from Kaggle...")
        
        # Try kagglehub first (easier, no authentication needed)
        success, download_path = download_with_kagglehub(DATASET_NAME)
        
        if success and download_path:
            # Copy from kagglehub cache to our data/raw
            if copy_kagglehub_dataset_to_raw(download_path, RAW_PATH):
                # Organize dataset
                if organize_dataset(RAW_PATH, TARGET_PATH):
                    print("\n" + "=" * 70)
                    print("SUCCESS! Dataset is ready for training")
                    print("Dataset has 3 classes: Minor, Moderate, Severe")
                    print("=" * 70)
            else:
                print("\nFailed to copy dataset from kagglehub cache")
                manual_download_instructions()
        else:
            # Try traditional Kaggle API
            success, download_path = download_with_kaggle_api(DATASET_NAME, RAW_PATH)
            
            if success:
                # Organize dataset
                if organize_dataset(RAW_PATH, TARGET_PATH):
                    print("\n" + "=" * 70)
                    print("SUCCESS! Dataset is ready for training")
                    print("Dataset has 3 classes: Minor, Moderate, Severe")
                    print("=" * 70)
            else:
                # Provide manual download instructions
                manual_download_instructions()
                print("\nAfter manual download, run this script again.")
    
    print("\n" + "=" * 70)
    print("Dataset preparation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("Verify the dataset structure in data/accident_images/")
    print("=" * 70)


if __name__ == "__main__":
    main()
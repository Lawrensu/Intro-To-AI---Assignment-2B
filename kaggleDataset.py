"""
Enhanced Dataset Preparation Script - Multi-Dataset Support
Downloads and organizes multiple Kaggle datasets for 4-class traffic incident classification

DATASETS:
1. Clean cars (none class) - kshitij192/cars-image-dataset
2. Minor damage - abdulrahmankerim/crash-car-image-hybrid-dataset-ccih
3. Moderate damage - marslanarshad/car-accidents-and-deformation-datasetannotated
4. Severe damage - exameese/accident-severity-image-dataset-v4
5. Additional mixed - prajwalbhamere/car-damage-severity-dataset

CLASSES: None, Minor, Moderate, Severe (4 classes)
"""

import os
import sys
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))


def download_dataset(dataset_id, description=""):
    """
    Download a dataset using kagglehub
    
    Args:
        dataset_id: Kaggle dataset identifier
        description: Human-readable description
    
    Returns:
        tuple: (success, path)
    """
    try:
        import kagglehub
        
        print(f"\n[DOWNLOADING] {description}")
        print(f"Dataset: {dataset_id}")
        print("Please wait...")
        
        path = kagglehub.dataset_download(dataset_id)
        print(f"[OK] Downloaded to: {path}")
        return True, path
    except Exception as e:
        print(f"[ERROR] Failed to download {dataset_id}: {e}")
        return False, None


def find_all_images_recursive(root_path, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Recursively find all images in a directory
    
    Args:
        root_path: Root directory to search
        extensions: Tuple of valid image extensions
    
    Returns:
        list: List of image file paths
    """
    images = []
    for ext in extensions:
        images.extend(root_path.rglob(f'*{ext}'))
        images.extend(root_path.rglob(f'*{ext.upper()}'))
    return images


def classify_and_copy_images(source_paths, target_class, raw_dir, batch_name=""):
    """
    Copy images from source to target class folder
    
    Args:
        source_paths: List of source directories or single directory
        target_class: Target class name (none/minor/moderate/severe)
        raw_dir: Raw backup directory
        batch_name: Optional batch identifier for naming
    
    Returns:
        int: Number of images copied
    """
    if not isinstance(source_paths, list):
        source_paths = [source_paths]
    
    target_dir = raw_dir / target_class
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    existing_count = len(list(target_dir.glob('*.jpg'))) + len(list(target_dir.glob('*.png')))
    
    for source_path in source_paths:
        source_path = Path(source_path)
        if not source_path.exists():
            print(f"[WARNING] Source path does not exist: {source_path}")
            continue
        
        # Find all images recursively
        images = find_all_images_recursive(source_path)
        
        if len(images) == 0:
            print(f"[WARNING] No images found in: {source_path}")
            continue
        
        print(f"  Found {len(images)} images in {source_path.name}")
        
        # Copy images with unique naming
        for i, img_path in enumerate(images):
            # Create unique filename
            if batch_name:
                new_name = f"{target_class}_{batch_name}_{existing_count + i:05d}{img_path.suffix}"
            else:
                new_name = f"{target_class}_{existing_count + i:05d}{img_path.suffix}"
            
            dest_path = target_dir / new_name
            
            try:
                shutil.copy2(img_path, dest_path)
                copied_count += 1
                
                # Progress indicator
                if (i + 1) % 500 == 0:
                    print(f"    Progress: {i+1}/{len(images)} images...")
            except Exception as e:
                print(f"[ERROR] Failed to copy {img_path.name}: {e}")
    
    return copied_count


def organize_into_splits(raw_dir, organized_dir, train_split=0.70, val_split=0.15, random_seed=42):
    """
    Organize raw images into train/val/test splits
    
    Args:
        raw_dir: Directory with raw images organized by class
        organized_dir: Target directory for organized dataset
        train_split: Training set proportion
        val_split: Validation set proportion
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: Statistics about the organization
    """
    print("\n" + "="*70)
    print("ORGANIZING INTO TRAIN/VAL/TEST SPLITS")
    print(f"Split ratios: {train_split:.0%} train / {val_split:.0%} val / {1-train_split-val_split:.0%} test")
    print("="*70)
    
    classes = ['none', 'minor', 'moderate', 'severe']
    stats = {}
    
    for class_name in classes:
        class_raw = raw_dir / class_name
        
        if not class_raw.exists():
            print(f"\n[WARNING] {class_name.upper()}: Directory not found, skipping")
            continue
        
        # Get all images
        images = list(class_raw.glob('*.jpg')) + list(class_raw.glob('*.png')) + \
                list(class_raw.glob('*.jpeg')) + list(class_raw.glob('*.bmp'))
        
        if len(images) == 0:
            print(f"\n[WARNING] {class_name.upper()}: No images found")
            continue
        
        print(f"\n[PROCESSING] {class_name.upper()}: {len(images)} images")
        
        # Shuffle and split
        import random
        random.seed(random_seed)
        random.shuffle(images)
        
        n_train = int(len(images) * train_split)
        n_val = int(len(images) * val_split)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy to organized structure
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        class_stats = {}
        
        for split_name, split_images in splits.items():
            split_dir = organized_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            copied = 0
            for i, img in enumerate(split_images):
                dest = split_dir / f"{class_name}_{split_name}_{i:05d}{img.suffix}"
                try:
                    shutil.copy2(img, dest)
                    copied += 1
                except Exception as e:
                    print(f"[ERROR] Failed to copy to {split_name}: {e}")
            
            class_stats[split_name] = copied
            print(f"  {split_name.upper()}: {copied} images copied")
        
        stats[class_name] = class_stats
    
    return stats


def print_final_summary(organized_dir, stats):
    """Print final dataset statistics"""
    print("\n" + "="*70)
    print("DATASET ORGANIZATION COMPLETE!")
    print("="*70)
    
    total_by_split = {'train': 0, 'val': 0, 'test': 0}
    total_by_class = {}
    
    print("\n[PER-CLASS DISTRIBUTION]")
    for class_name in ['none', 'minor', 'moderate', 'severe']:
        if class_name in stats:
            class_stats = stats[class_name]
            train_count = class_stats.get('train', 0)
            val_count = class_stats.get('val', 0)
            test_count = class_stats.get('test', 0)
            total_class = train_count + val_count + test_count
            
            total_by_class[class_name] = total_class
            total_by_split['train'] += train_count
            total_by_split['val'] += val_count
            total_by_split['test'] += test_count
            
            print(f"  {class_name.upper():>10s}: Train={train_count:4d} | Val={val_count:4d} | Test={test_count:4d} | Total={total_class:4d}")
    
    total_images = sum(total_by_class.values())
    
    print("\n[BY SPLIT]")
    for split in ['train', 'val', 'test']:
        count = total_by_split[split]
        pct = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {split.upper():>5s}: {count:4d} images ({pct:.1f}%)")
    
    print(f"\n[TOTAL]: {total_images} images across 4 classes")
    
    # Quality assessment
    print("\n[QUALITY ASSESSMENT]")
    if total_images >= 8000:
        grade = "A+ [EXCELLENT - Top-tier dataset]"
    elif total_images >= 6000:
        grade = "A [EXCELLENT]"
    elif total_images >= 4000:
        grade = "B+ [VERY GOOD]"
    elif total_images >= 2500:
        grade = "B [GOOD]"
    else:
        grade = "C [ACCEPTABLE but could use more data]"
    
    print(f"  Dataset Size: {grade}")
    
    # Balance check
    if len(total_by_class) == 4:
        max_class = max(total_by_class.values())
        min_class = min(total_by_class.values())
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        if imbalance_ratio <= 2:
            balance = "WELL BALANCED"
        elif imbalance_ratio <= 3:
            balance = "MODERATELY BALANCED"
        else:
            balance = "IMBALANCED (class weights recommended)"
        
        print(f"  Class Balance: {balance} (ratio: {imbalance_ratio:.2f}:1)")
    
    print("\n[SAVED TO]")
    print(f"  Organized: {organized_dir.absolute()}")
    print(f"  Raw backup: {organized_dir.parent / 'raw'}")


def main():
    print("="*70)
    print("MULTI-DATASET PREPARATION FOR 4-CLASS INCIDENT CLASSIFICATION")
    print("Classes: None, Minor, Moderate, Severe")
    print("="*70)
    
    # Paths
    RAW_DIR = Path("data/raw")
    ORGANIZED_DIR = Path("data/accident_images")
    
    # Check if organized dataset already exists
    if ORGANIZED_DIR.exists() and len(list(ORGANIZED_DIR.rglob('*.jpg'))) > 0:
        print(f"\n[INFO] Organized dataset already exists")
        
        # Count existing images
        train_count = len(list((ORGANIZED_DIR / 'train').rglob('*.jpg')))
        val_count = len(list((ORGANIZED_DIR / 'val').rglob('*.jpg')))
        test_count = len(list((ORGANIZED_DIR / 'test').rglob('*.jpg')))
        total = train_count + val_count + test_count
        
        print(f"  Train: {train_count} images")
        print(f"  Val: {val_count} images")
        print(f"  Test: {test_count} images")
        print(f"  TOTAL: {total} images")
        
        response = input("\nRe-download and reorganize ALL datasets? (y/n): ")
        if response.lower() != 'y':
            print("[INFO] Using existing dataset")
            return
        
        # Clean up
        print("\n[INFO] Removing old data...")
        if RAW_DIR.exists():
            shutil.rmtree(RAW_DIR)
        if ORGANIZED_DIR.exists():
            shutil.rmtree(ORGANIZED_DIR)
        time.sleep(1)
    
    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("DOWNLOADING DATASETS FROM KAGGLE")
    print("This will download ~5-8 GB of data (may take 10-30 minutes)")
    print("="*70)
    
    # Dataset definitions
    datasets = [
        {
            'id': 'kshitij192/cars-image-dataset',
            'class': 'none',
            'description': 'Clean Cars Dataset (No Damage)',
            'batch': 'clean'
        },
        {
            'id': 'prajwalbhamere/car-damage-severity-dataset',
            'class': 'mixed',  # Will be split by folder
            'description': 'Car Damage Severity Dataset (Mixed)',
            'batch': 'severity'
        },
        {
            'id': 'abdulrahmankerim/crash-car-image-hybrid-dataset-ccih',
            'class': 'minor',
            'description': 'Crash Car Hybrid Dataset (Minor)',
            'batch': 'hybrid'
        },
        {
            'id': 'marslanarshad/car-accidents-and-deformation-datasetannotated',
            'class': 'moderate',
            'description': 'Car Accidents & Deformation Dataset (Moderate)',
            'batch': 'deform'
        },
        {
            'id': 'exameese/accident-severity-image-dataset-v4',
            'class': 'severe',
            'description': 'Accident Severity Dataset v4 (Severe)',
            'batch': 'severity_v4'
        }
    ]
    
    # Download and organize each dataset
    downloaded_paths = []
    
    for idx, dataset in enumerate(datasets, 1):
        print(f"\n[{idx}/{len(datasets)}] " + "="*50)
        success, path = download_dataset(dataset['id'], dataset['description'])
        
        if success and path:
            downloaded_paths.append({
                'path': path,
                'class': dataset['class'],
                'batch': dataset['batch']
            })
        else:
            print(f"[WARNING] Skipping {dataset['description']}")
        
        time.sleep(1)  # Brief pause between downloads
    
    if len(downloaded_paths) == 0:
        print("\n[ERROR] No datasets were downloaded successfully!")
        print("Please check your internet connection and Kaggle API setup")
        return
    
    # Organize downloaded datasets into class folders
    print("\n" + "="*70)
    print("ORGANIZING DOWNLOADED DATASETS INTO CLASS FOLDERS")
    print("="*70)
    
    total_copied = {'none': 0, 'minor': 0, 'moderate': 0, 'severe': 0}
    
    for dl in downloaded_paths:
        path = Path(dl['path'])
        class_type = dl['class']
        batch = dl['batch']
        
        print(f"\n[PROCESSING] {batch} dataset")
        
        if class_type == 'mixed':
            # Handle prajwalbhamere dataset (has folder structure)
            print("  Detecting class folders...")
            
            # Find class-specific folders
            minor_paths = []
            moderate_paths = []
            severe_paths = []
            
            for item in path.rglob('*'):
                if item.is_dir():
                    folder_name = item.name.lower()
                    
                    # Check if folder has images
                    img_count = len(find_all_images_recursive(item))
                    if img_count < 10:
                        continue
                    
                    # Classify folder
                    if any(kw in folder_name for kw in ['01', 'minor', 'light']):
                        minor_paths.append(item)
                        print(f"    Found MINOR: {item.name} ({img_count} images)")
                    elif any(kw in folder_name for kw in ['02', 'moderate', 'medium']):
                        moderate_paths.append(item)
                        print(f"    Found MODERATE: {item.name} ({img_count} images)")
                    elif any(kw in folder_name for kw in ['03', 'severe', 'heavy']):
                        severe_paths.append(item)
                        print(f"    Found SEVERE: {item.name} ({img_count} images)")
            
            # Copy to respective classes
            if minor_paths:
                print(f"\n  Copying to MINOR class...")
                count = classify_and_copy_images(minor_paths, 'minor', RAW_DIR, batch)
                total_copied['minor'] += count
                print(f"    [OK] Copied {count} images")
            
            if moderate_paths:
                print(f"\n  Copying to MODERATE class...")
                count = classify_and_copy_images(moderate_paths, 'moderate', RAW_DIR, batch)
                total_copied['moderate'] += count
                print(f"    [OK] Copied {count} images")
            
            if severe_paths:
                print(f"\n  Copying to SEVERE class...")
                count = classify_and_copy_images(severe_paths, 'severe', RAW_DIR, batch)
                total_copied['severe'] += count
                print(f"    [OK] Copied {count} images")
        
        else:
            # Single-class dataset
            print(f"  Copying to {class_type.upper()} class...")
            count = classify_and_copy_images(path, class_type, RAW_DIR, batch)
            total_copied[class_type] += count
            print(f"    [OK] Copied {count} images")
    
    # Summary of raw organization
    print("\n" + "="*70)
    print("RAW DATASET ORGANIZATION SUMMARY")
    print("="*70)
    for class_name, count in total_copied.items():
        print(f"  {class_name.upper():>10s}: {count:5d} images")
    print(f"  {'TOTAL':>10s}: {sum(total_copied.values()):5d} images")
    
    # Organize into train/val/test splits
    stats = organize_into_splits(RAW_DIR, ORGANIZED_DIR)
    
    # Print final summary
    print_final_summary(ORGANIZED_DIR, stats)
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\n[NEXT STEPS]")
    print("  1. python src/data_processing.py  # Verify dataset")
    print("  2. python src/train_cnn.py        # Train CNN model")
    print("\n[EXPECTED RESULTS]")
    print("  - With 6000+ images: 90-95% accuracy achievable")
    print("  - With 8000+ images: 92-96% accuracy achievable")
    print("  - Training time: 20-40 minutes (GPU) / 4-6 hours (CPU)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Dataset preparation cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
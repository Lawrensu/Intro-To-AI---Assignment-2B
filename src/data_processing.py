"""
Data processing for 4-class Traffic Incident Classification
Classes: None, Minor, Moderate, Severe
"""

import torch
from pathlib import Path
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Import transforms separately to avoid circular import
import torchvision.transforms as transforms
import torchvision.datasets as datasets

CLASS_NAMES = ['none', 'minor', 'moderate', 'severe']
NUM_CLASSES = 4
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_data_transforms(augment=True):
    """Get transforms for train/val/test"""
    if augment:
        return {
            'train': transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ]),
            'val': transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ]),
            'test': transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
        }
    else:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        return {'train': transform, 'val': transform, 'test': transform}


def create_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4,
                       augment: bool = True) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    """Create dataloaders for train/val/test"""
    data_dir = Path(data_dir)
    data_transforms = get_data_transforms(augment)
    
    # Create datasets using ImageFolder
    image_datasets = {
        split: datasets.ImageFolder(
            root=str(data_dir / split),
            transform=data_transforms[split]
        )
        for split in ['train', 'val', 'test']
    }
    
    # Verify classes
    actual_classes = sorted(image_datasets['train'].classes)
    expected_classes = sorted(CLASS_NAMES)
    
    print(f"\n[OK] Found classes: {actual_classes}")
    print(f"     Expected: {expected_classes}")
    
    if actual_classes != expected_classes:
        print(f"[WARNING] Class mismatch!")
        print(f"          This may cause issues during training")
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            image_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders, image_datasets


def calculate_class_weights(data_dir: str, split: str = 'train') -> torch.Tensor:
    """Calculate class weights for handling class imbalance"""
    data_dir = Path(data_dir) / split
    
    class_counts = []
    for class_name in CLASS_NAMES:
        class_path = data_dir / class_name
        if class_path.exists():
            count = len(list(class_path.glob('*.jpg'))) + len(list(class_path.glob('*.png')))
            class_counts.append(count)
        else:
            class_counts.append(0)
    
    total_samples = sum(class_counts)
    if total_samples == 0:
        return torch.ones(NUM_CLASSES)
    
    class_weights = [
        total_samples / (NUM_CLASSES * count) if count > 0 else 0
        for count in class_counts
    ]
    
    return torch.FloatTensor(class_weights)


def evaluate_dataset_sufficiency(data_dir: str) -> Dict:
    """Check if dataset is sufficient"""
    data_dir = Path(data_dir)
    stats = {
        'sufficient': True,
        'warnings': [],
        'recommendations': [],
        'total_images': 0
    }
    
    for split in ['train', 'val', 'test']:
        split_path = data_dir / split
        if not split_path.exists():
            stats['sufficient'] = False
            stats['warnings'].append(f"Missing {split} directory")
            continue
        
        split_counts = {}
        for class_name in CLASS_NAMES:
            class_path = split_path / class_name
            if class_path.exists():
                count = len(list(class_path.glob('*.jpg'))) + len(list(class_path.glob('*.png')))
                split_counts[class_name] = count
            else:
                split_counts[class_name] = 0
                stats['warnings'].append(f"Missing {class_name} in {split}")
        
        stats[split] = split_counts
        stats['total_images'] += sum(split_counts.values())
        
        # Check minimum samples
        min_samples = {'train': 100, 'val': 20, 'test': 20}
        for class_name, count in split_counts.items():
            if count < min_samples[split]:
                stats['sufficient'] = False
                stats['warnings'].append(
                    f"{class_name} in {split} has only {count} samples (need >= {min_samples[split]})"
                )
    
    # Add recommendations
    total = stats['total_images']
    if total < 500:
        stats['recommendations'].append("[WARNING] Dataset is VERY small - use heavy data augmentation")
    elif total < 1000:
        stats['recommendations'].append("[WARNING] Dataset is small - use data augmentation")
    elif total < 2000:
        stats['recommendations'].append("[OK] Dataset size is acceptable")
    else:
        stats['recommendations'].append("[OK] Dataset size is GOOD")
    
    return stats


# Test module
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING DATA PROCESSING MODULE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    data_dir = "data/accident_images"
    
    if not Path(data_dir).exists():
        print(f"\n[ERROR] Dataset not found at: {data_dir}")
        print("\n[INFO] Run first: python kaggleDataset.py")
        exit(1)
    
    # Evaluate dataset
    print("\n" + "=" * 70)
    print("EVALUATING DATASET")
    print("=" * 70)
    
    stats = evaluate_dataset_sufficiency(data_dir)
    
    print(f"\n[SUMMARY] Dataset Statistics:")
    for split in ['train', 'val', 'test']:
        if split in stats:
            total = sum(stats[split].values())
            print(f"\n{split.capitalize()} set: {total} images")
            for class_name, count in stats[split].items():
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {class_name:>10s}: {count:4d} images ({pct:5.1f}%)")
    
    print(f"\nTotal images: {stats['total_images']}")
    
    if stats['warnings']:
        print("\n[WARNINGS]")
        for w in stats['warnings']:
            print(f"  - {w}")
    
    if stats['recommendations']:
        print("\n[RECOMMENDATIONS]")
        for r in stats['recommendations']:
            print(f"  - {r}")
    
    if stats['sufficient']:
        print("\n[OK] Dataset is sufficient!")
        
        # Test dataloader
        print("\n" + "=" * 70)
        print("TESTING DATALOADER")
        print("=" * 70)
        
        try:
            dataloaders, datasets_dict = create_dataloaders(
                data_dir,
                batch_size=16,
                num_workers=0,  # Use 0 for testing
                augment=True
            )
            
            print("[OK] Dataloader created successfully!")
            
            # Get a batch
            images, labels = next(iter(dataloaders['train']))
            print(f"\n[SAMPLE] Batch:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Unique labels: {torch.unique(labels).tolist()}")
            print(f"  Label distribution: {torch.bincount(labels).tolist()}")
            
            print("\n[OK] Data processing test PASSED!")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n[ERROR] Dataloader test FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[ERROR] Dataset is NOT sufficient!")
        print("        Fix the warnings above before training")
    
    print("=" * 70)
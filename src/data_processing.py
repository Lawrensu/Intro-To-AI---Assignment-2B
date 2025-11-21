"""
Data processing utilities for the Traffic Incident Classification System

Handles dataset preparation, augmentation, and loading for 3-class classification:
- Class 0: Minor
- Class 1: Moderate  
- Class 2: Severe

Note: No "none" class - we only classify existing traffic incidents
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image


# Class mapping for 3-class incident severity classification
CLASS_NAMES = ['minor', 'moderate', 'severe']
NUM_CLASSES = 3

# Image preprocessing constants
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_data_transforms(augment=True):
    """
    Get data transformation pipelines for training and validation
    
    Args:
        augment (bool): Whether to apply data augmentation for training
    
    Returns:
        dict: Dictionary containing 'train' and 'val' transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def create_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, 
                       augment: bool = True) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    """
    Create PyTorch dataloaders for train, validation, and test sets
    
    Args:
        data_dir (str): Path to the organized dataset directory
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        augment (bool): Whether to apply data augmentation
    
    Returns:
        tuple: (dataloaders_dict, datasets_dict)
    """
    data_transforms = get_data_transforms(augment=augment)
    
    # Create datasets for each split
    image_datasets = {
        split: datasets.ImageFolder(
            root=os.path.join(data_dir, split),
            transform=data_transforms[split]
        )
        for split in ['train', 'val', 'test']
    }
    
    # Verify we have exactly 3 classes
    for split, dataset in image_datasets.items():
        if len(dataset.classes) != NUM_CLASSES:
            print(f"WARNING: {split} set has {len(dataset.classes)} classes, expected {NUM_CLASSES}")
            print(f"Classes found: {dataset.classes}")
    
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
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("=" * 70)
    for split in ['train', 'val', 'test']:
        dataset = image_datasets[split]
        print(f"{split.capitalize()} set: {len(dataset)} images")
        
        # Count images per class
        class_counts = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_name = dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} images")
    print("=" * 70)
    
    return dataloaders, image_datasets


def prepare_kaggle_dataset(kaggle_dir: str, target_dir: str, 
                           train_split: float = 0.7, val_split: float = 0.15,
                           random_seed: int = 42):
    """
    Organize Kaggle dataset into train/val/test splits
    
    Maps from raw folder names to training folder names:
    - minor_damage -> minor (Class 0)
    - moderate_damage -> moderate (Class 1)
    - severe_damage -> severe (Class 2)
    
    Args:
        kaggle_dir (str): Path to raw Kaggle dataset (data/raw/)
        target_dir (str): Path to organized dataset (data/accident_images/)
        train_split (float): Proportion of data for training (default: 0.7)
        val_split (float): Proportion of data for validation (default: 0.15)
        random_seed (int): Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Folder name mapping (raw -> organized)
    folder_mapping = {
        'minor_damage': 'minor',
        'moderate_damage': 'moderate',
        'severe_damage': 'severe'
    }
    
    print(f"\nPreparing dataset from {kaggle_dir} to {target_dir}...")
    print(f"Split: Train={train_split:.0%}, Val={val_split:.0%}, Test={1-train_split-val_split:.0%}")
    
    # Create target directory structure
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)
    
    # Process each raw class folder
    total_images = 0
    class_stats = {class_name: {'train': 0, 'val': 0, 'test': 0} for class_name in CLASS_NAMES}
    
    for raw_folder, target_folder in folder_mapping.items():
        source_dir = os.path.join(kaggle_dir, raw_folder)
        
        if not os.path.exists(source_dir):
            print(f"WARNING: {source_dir} not found, skipping...")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(source_dir).glob(ext))
        
        if len(image_files) == 0:
            print(f"WARNING: No images found in {source_dir}")
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        n_train = int(len(image_files) * train_split)
        n_val = int(len(image_files) * val_split)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to respective splits
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split, files in splits.items():
            dest_dir = os.path.join(target_dir, split, target_folder)
            for img_file in files:
                dest_path = os.path.join(dest_dir, img_file.name)
                shutil.copy2(img_file, dest_path)
                class_stats[target_folder][split] += 1
                total_images += 1
        
        print(f"  {raw_folder} -> {target_folder}: "
              f"{len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Print final statistics
    print("\nDataset Organization Complete!")
    print("=" * 70)
    print(f"Total images processed: {total_images}")
    print("\nClass distribution:")
    for class_name in CLASS_NAMES:
        stats = class_stats[class_name]
        total = sum(stats.values())
        print(f"  {class_name.capitalize()}: {total} total "
              f"(train: {stats['train']}, val: {stats['val']}, test: {stats['test']})")
    print("=" * 70)


def calculate_class_weights(data_dir: str, split: str = 'train') -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        data_dir (str): Path to organized dataset
        split (str): Which split to calculate weights from (default: 'train')
    
    Returns:
        torch.Tensor: Class weights for loss function
    """
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, split))
    
    # Count samples per class
    class_counts = [0] * NUM_CLASSES
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Calculate inverse frequency weights
    total = sum(class_counts)
    weights = [total / (NUM_CLASSES * count) if count > 0 else 0 for count in class_counts]
    
    print("\nClass Weights (for handling imbalance):")
    for i, (class_name, weight, count) in enumerate(zip(CLASS_NAMES, weights, class_counts)):
        print(f"  Class {i} ({class_name}): {count} samples, weight: {weight:.4f}")
    
    return torch.FloatTensor(weights)


def get_sample_images(data_dir: str, split: str = 'train', num_samples: int = 5):
    """
    Get sample images from each class for visualization
    
    Args:
        data_dir (str): Path to organized dataset
        split (str): Which split to sample from
        num_samples (int): Number of samples per class
    
    Returns:
        dict: Dictionary mapping class names to lists of image paths
    """
    samples = {class_name: [] for class_name in CLASS_NAMES}
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, split, class_name)
        if os.path.exists(class_dir):
            image_files = list(Path(class_dir).glob('*.jpg')) + list(Path(class_dir).glob('*.png'))
            samples[class_name] = random.sample(image_files, min(num_samples, len(image_files)))
    
    return samples


class IncidentDataset(Dataset):
    """
    Custom Dataset for Traffic Incident Classification
    """
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        """
        Args:
            data_dir (str): Path to organized dataset
            split (str): 'train', 'val', or 'test'
            transform: Torchvision transforms to apply
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Utility function to denormalize images for visualization
def denormalize(tensor, mean=MEAN, std=STD):
    """
    Denormalize tensor for visualization
    
    Args:
        tensor: Normalized image tensor
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


if __name__ == "__main__":
    # Test data processing
    print("Testing data processing module...")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class names: {CLASS_NAMES}")
    
    # Test if dataset exists
    data_dir = "data/accident_images"
    if os.path.exists(data_dir):
        try:
            dataloaders, datasets_dict = create_dataloaders(data_dir, batch_size=16)
            print("\n✓ Data loading successful!")
            
            # Test batch loading
            images, labels = next(iter(dataloaders['train']))
            print(f"\nBatch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Unique labels in batch: {labels.unique().tolist()}")
            
        except Exception as e:
            print(f"\n✗ Error loading data: {e}")
    else:
        print(f"\n⚠ Dataset not found at {data_dir}")
        print("Run 'python kaggleDataset.py' to prepare the dataset first")
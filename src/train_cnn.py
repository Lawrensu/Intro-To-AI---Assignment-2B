"""
Training script for CNN model
Run this to train the ResNet-18 based incident classifier
"""

import torch
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import create_dataloaders, evaluate_dataset_sufficiency
from src.models.cnn_model import IncidentCNN, CNNTrainer


def get_class_distribution(dataset):
    """Get class distribution from ImageFolder dataset"""
    targets = dataset.targets
    unique, counts = np.unique(targets, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def main():
    # Configuration
    DATA_DIR = "data/accident_images"
    MODEL_SAVE_PATH = "models/cnn_model.pth"
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("Training CNN Model for Traffic Incident Classification")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 70)
    
    # Evaluate dataset
    print("\nStep 1: Evaluating dataset...")
    analysis = evaluate_dataset_sufficiency(DATA_DIR)
    print(f"Total images: {analysis['total_images']}")
    if analysis['recommendations']:
        print("Recommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
    
    # Create dataloaders
    print("\nStep 2: Creating dataloaders...")
    dataloaders, datasets = create_dataloaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE, 
        num_workers=4, 
        augment=True
    )
    
    # Get class distribution for weighting
    train_dist = get_class_distribution(datasets['train'])
    class_counts = [train_dist.get(i, 0) for i in range(4)]
    
    print(f"\n[INFO] Class distribution in training set:")
    for i, count in enumerate(class_counts):
        class_name = datasets['train'].classes[i]
        print(f"  {class_name:>10s}: {count:4d} images ({count/sum(class_counts)*100:.1f}%)")
    
    # Create model
    print("\nStep 3: Creating model...")
    model = IncidentCNN(num_classes=4, pretrained=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[OK] Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture: ResNet-18")
    print(f"  Number of classes: 4")
    
    # Create trainer
    trainer = CNNTrainer(model, device=DEVICE)
    
    # Calculate class weights for imbalanced dataset
    class_weights = None
    if max(class_counts) / min(class_counts) > 2:
        print("\n[INFO] Dataset is imbalanced. Calculating class weights...")
        total_samples = sum(class_counts)
        class_weights = torch.FloatTensor([
            total_samples / (4 * count) if count > 0 else 0 
            for count in class_counts
        ])
        print(f"  Class weights: {class_weights.numpy()}")
    
    # Train model
    print("\nStep 4: Training model...")
    print("=" * 70)
    
    history = trainer.train(
        dataloaders=dataloaders,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        patience=5,
        save_path=MODEL_SAVE_PATH
    )
    
    # Evaluate on test set
    print("\nStep 5: Evaluating on test set...")
    results = trainer.evaluate(dataloaders['test'])
    
    print(f"\n[RESULTS] Test Set Performance:")
    print(f"  Overall Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    
    print(f"\n[PER-CLASS] Performance:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"\n  {class_name.upper()}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1_score']:.4f}")
    
    # Save confusion matrix
    print("\nStep 6: Generating visualizations...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=datasets['test'].classes,
                yticklabels=datasets['test'].classes)
    plt.title('CNN Confusion Matrix (Test Set)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('models/cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("[OK] Confusion matrix saved to: models/cnn_confusion_matrix.png")
    plt.close()
    
    # Save training history plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/cnn_training_history.png', dpi=300, bbox_inches='tight')
    print("[OK] Training history saved to: models/cnn_training_history.png")
    plt.close()
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n[SAVED FILES]")
    print(f"  Model checkpoint: {MODEL_SAVE_PATH}")
    print(f"  Training curves: models/cnn_training_history.png")
    print(f"  Confusion matrix: models/cnn_confusion_matrix.png")
    
    print(f"\n[FINAL RESULTS]")
    print(f"  Best validation accuracy: {trainer.best_acc:.4f}")
    print(f"  Test accuracy: {results['accuracy']:.4f}")
    print(f"  Total training time: {sum(history.get('epoch_times', [0])):.1f} seconds")
    
    print("\n" + "=" * 70)
    print("[NEXT STEPS]")
    print("  1. Check training curves: models/cnn_training_history.png")
    print("  2. Review confusion matrix: models/cnn_confusion_matrix.png")
    print("  3. Use model for predictions with src/models/cnn_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
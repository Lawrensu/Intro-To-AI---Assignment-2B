"""
Training script for CNN model
Run this to train the ResNet-18 based incident classifier
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import create_dataloaders, evaluate_dataset_sufficiency
from src.models.cnn_model import IncidentCNN, CNNTrainer


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
    train_dist = datasets['train'].get_class_distribution()
    class_counts = [train_dist.get(i, 0) for i in range(4)]
    
    # Create model
    print("\nStep 3: Creating model...")
    model = IncidentCNN(num_classes=4, pretrained=True, freeze_backbone=False)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create trainer
    trainer = CNNTrainer(model, device=DEVICE, learning_rate=LEARNING_RATE)
    
    # Set class weights if dataset is imbalanced
    if max(class_counts) / min(class_counts) > 2:
        print("\nDataset is imbalanced. Setting class weights...")
        trainer.set_class_weights(class_counts)
    
    # Train model
    print("\nStep 4: Training model...")
    history = trainer.train(
        dataloaders['train'], 
        dataloaders['val'], 
        num_epochs=NUM_EPOCHS,
        save_path=MODEL_SAVE_PATH
    )
    
    # Plot training history
    print("\nStep 5: Plotting training history...")
    trainer.plot_training_history('models/cnn_training_history.png')
    
    # Evaluate on test set
    print("\nStep 6: Evaluating on test set...")
    results = trainer.evaluate(dataloaders['test'])
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(
        results['confusion_matrix'], 
        datasets['test'].classes,
        'models/cnn_confusion_matrix.png'
    )
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
"""
Train 3 Image Classification Models for Traffic Incident Severity
Models: ResNet-18, MobileNet-V2, EfficientNet-B0
Author: Team (Lawrence, Faridz, Cherylynn, Jason)
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import create_dataloaders, evaluate_dataset_sufficiency
from src.models.cnn_model import CNNTrainer

# Configuration
DATA_DIR = "data/accident_images"
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_CLASSES = 4
CLASS_NAMES = ['none', 'minor', 'moderate', 'severe']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model 1: ResNet-18 (Baseline - Skip Connections)
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Model 2: MobileNet-V2 (Efficient - Depthwise Separable Convolutions)
class MobileNetV2Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Model 3: EfficientNet-B0 (State-of-the-art - Compound Scaling)
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(model, model_name, model_description, save_path, dataloaders, datasets):
    """Train a single model"""
    print("\n" + "="*70)
    print(f"TRAINING {model_name}")
    print(f"Architecture: {model_description}")
    print("="*70)
    
    model = model.to(DEVICE)
    
    # Create trainer
    trainer = CNNTrainer(model=model, device=DEVICE)
    
    # Calculate class weights
    from collections import Counter
    train_labels = [label for _, label in datasets['train']]
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    class_weights = torch.FloatTensor([
        total / (NUM_CLASSES * class_counts[i]) if i in class_counts else 0
        for i in range(NUM_CLASSES)
    ])
    
    # Train
    history = trainer.train(
        dataloaders=dataloaders,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights,
        patience=5,
        save_path=save_path
    )
    
    # Evaluate
    print(f"\nEvaluating {model_name} on test set...")
    results = trainer.evaluate(dataloaders['test'], class_names=CLASS_NAMES)
    
    print(f"\n[RESULTS] {model_name} Test Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    
    # Save visualizations
    save_training_plots(history, model_name)
    save_confusion_matrix(results, model_name, CLASS_NAMES)
    
    print(f"\n[OK] {model_name} complete!")
    print(f"  Best accuracy: {trainer.best_acc:.4f}")
    print(f"  Model saved: {save_path}")
    
    return model, history, results

def save_training_plots(history, model_name):
    """Save training history plots"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='#e74c3c')
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, color='#3498db')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', linewidth=2, color='#e74c3c')
    plt.plot(history['val_acc'], label='Val Acc', linewidth=2, color='#3498db')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{model_name} - Training Accuracy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] Learning curves: models/{model_name}_learning_curves.png")
    plt.close()

def save_confusion_matrix(results, model_name, class_names):
    """Save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] Confusion matrix: models/{model_name}_confusion_matrix.png")
    plt.close()

def compare_models(results_dict):
    """Compare all 3 models"""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score']
        })
    
    # Print comparison table
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for data in comparison_data:
        print(f"{data['Model']:<20} {data['Accuracy']:<12.4f} "
              f"{data['Precision']:<12.4f} {data['Recall']:<12.4f} {data['F1-Score']:<12.4f}")
    
    # Determine best model
    best_model = max(comparison_data, key=lambda x: x['Accuracy'])
    print(f"\n[BEST MODEL] {best_model['Model']} ({best_model['Accuracy']:.2%} accuracy)")
    
    # Architecture comparison
    print("\n" + "="*70)
    print("ARCHITECTURE DIFFERENCES")
    print("="*70)
    print("""
ResNet-18:
  - Skip connections (residual learning)
  - 18 layers deep
  - Parameters: 11M
  - Best for: General-purpose image classification
  
MobileNet-V2:
  - Depthwise separable convolutions
  - Inverted residuals with linear bottlenecks
  - Parameters: 3.5M
  - Best for: Mobile/edge deployment, real-time inference
  
EfficientNet-B0:
  - Compound scaling (depth + width + resolution)
  - Neural architecture search optimized
  - Parameters: 5M
  - Best for: State-of-the-art accuracy with efficiency
    """)
    
    # Save comparison plot
    plt.figure(figsize=(12, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(comparison_data))
    width = 0.2
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, metric in enumerate(metrics):
        values = [d[metric] for d in comparison_data]
        plt.bar(x + i * width, values, width, label=metric, color=colors[i])
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Comparison - All Metrics', fontsize=14, fontweight='bold')
    plt.xticks(x + width * 1.5, [d['Model'] for d in comparison_data])
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Comparison chart: models/model_comparison.png")
    plt.close()

def main():
    print("="*70)
    print("TRAINING 3 IMAGE CLASSIFICATION MODELS")
    print("ResNet-18 + MobileNet-V2 + EfficientNet-B0")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    Path("models").mkdir(exist_ok=True)
    
    # Evaluate dataset
    print("\nChecking dataset...")
    analysis = evaluate_dataset_sufficiency(DATA_DIR)
    print(f"Total images: {analysis['total_images']}")
    if analysis['recommendations']:
        print("Dataset recommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
    
    # Create dataloaders (shared across all models)
    print("\nCreating dataloaders...")
    dataloaders, datasets = create_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4,
        augment=True
    )
    
    print(f"Train: {len(datasets['train'])} images")
    print(f"Val: {len(datasets['val'])} images")
    print(f"Test: {len(datasets['test'])} images")
    
    # Train all 3 models
    results_dict = {}
    
    # Model 1: ResNet-18
    print("\n" + "="*70)
    print("MODEL 1/3: ResNet-18")
    print("="*70)
    resnet18 = ResNet18Model(num_classes=NUM_CLASSES)
    _, _, results1 = train_model(
        resnet18,
        "ResNet18",
        "Residual Networks with skip connections",
        "models/resnet18_model.pth",
        dataloaders,
        datasets
    )
    results_dict['ResNet-18'] = results1
    
    # Model 2: MobileNet-V2
    print("\n" + "="*70)
    print("MODEL 2/3: MobileNet-V2")
    print("="*70)
    mobilenet = MobileNetV2Model(num_classes=NUM_CLASSES)
    _, _, results2 = train_model(
        mobilenet,
        "MobileNetV2",
        "Depthwise separable convolutions for efficiency",
        "models/mobilenet_model.pth",
        dataloaders,
        datasets
    )
    results_dict['MobileNet-V2'] = results2
    
    # Model 3: EfficientNet
    print("\n" + "="*70)
    print("MODEL 3/3: EfficientNet-B0")
    print("="*70)
    efficientnet = EfficientNetModel(num_classes=NUM_CLASSES)
    _, _, results3 = train_model(
        efficientnet,
        "EfficientNet-B0",
        "Compound scaling for optimal performance",
        "models/efficientnet_model.pth",
        dataloaders,
        datasets
    )
    results_dict['EfficientNet-B0'] = results3
    
    # Compare all models
    compare_models(results_dict)
    
    print("\n" + "="*70)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("="*70)
    print("\nOutput files:")
    print("  Models:")
    print("    - models/resnet18_model.pth")
    print("    - models/mobilenet_model.pth")
    print("    - models/efficientnet_model.pth")
    print("\n  Learning Curves (3 files):")
    print("    - models/ResNet18_learning_curves.png")
    print("    - models/MobileNetV2_learning_curves.png")
    print("    - models/EfficientNet-B0_learning_curves.png")
    print("\n  Confusion Matrices (3 files):")
    print("    - models/ResNet18_confusion_matrix.png")
    print("    - models/MobileNetV2_confusion_matrix.png")
    print("    - models/EfficientNet-B0_confusion_matrix.png")
    print("\n  Comparison:")
    print("    - models/model_comparison.png")

if __name__ == "__main__":
    main()
"""
Training script for GCN model
Run this to train the Graph Convolutional Network for traffic flow prediction
Author: Cherylynn
"""

import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gcn_model import TrafficGCN, GCNTrainer
from src.graph_construction import parse_road_network, construct_graph


# Configuration
NETWORK_FILE = "heritage_assignment_15_time_asymmetric-1.txt"
MODEL_SAVE_PATH = "models/gcn_model.pth"
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
PATIENCE = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_training_history(history, save_path='models/gcn_training_history.png'):
    """
    Plot and save training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2, color='#2E86AB')
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2, color='#A23B72')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history saved to {save_path}")
    plt.close()


def plot_confusion_matrix(data, model, mask, class_names, save_path='models/gcn_confusion_matrix.png'):
    """
    Plot and save confusion matrix
    
    Args:
        data: PyG Data object
        model: Trained GCN model
        mask: Test mask
        class_names: List of class names
        save_path: Path to save plot
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    plt.title('GCN Confusion Matrix (Test Set)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()
    
    # Print classification report
    print("\n" + "=" * 70)
    print("Classification Report (Test Set)")
    print("=" * 70)
    # Get unique labels present in test set
    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names_present = [class_names[i] for i in labels_present]
    print(classification_report(y_true, y_pred, labels=labels_present, target_names=target_names_present, digits=4))


def calculate_metrics(data, model, mask):
    """
    Calculate detailed evaluation metrics
    
    Args:
        data: PyG Data object
        model: Trained model
        mask: Evaluation mask
    
    Returns:
        dict: Metrics dictionary with labels
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get labels present in predictions and ground truth
    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_present, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'labels_present': labels_present
    }
    
    return metrics


def print_evaluation_summary(metrics, class_names):
    """
    Print formatted evaluation summary
    
    Args:
        metrics: Metrics dictionary
        class_names: List of class names
    """
    print("\n" + "=" * 70)
    print("GCN Model Evaluation Summary")
    print("=" * 70)
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 70)
    
    # Only print metrics for classes present in test set
    labels_present = metrics.get('labels_present', range(len(class_names)))
    for idx, label_idx in enumerate(labels_present):
        class_name = class_names[label_idx]
        print(f"{class_name:<20} {metrics['precision'][idx]:<15.4f} "
              f"{metrics['recall'][idx]:<15.4f} {metrics['f1_score'][idx]:<15.4f}")
    
    print("-" * 70)


def main():
    """Main training function"""
    print("=" * 70)
    print("GCN Model Training for Traffic Flow Prediction")
    print("COS30019 - Introduction to AI")
    print("Author: Cherrylynn")
    print("=" * 70)
    
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Step 1: Load and construct graph
    print("\nStep 1: Loading road network and constructing graph...")
    if not Path(NETWORK_FILE).exists():
        print(f"Error: Network file not found: {NETWORK_FILE}")
        print("Please ensure the network file is in the project root directory.")
        return
    
    nodes, ways, cameras, meta = parse_road_network(NETWORK_FILE)
    data = construct_graph(nodes, ways)
    
    # Step 2: Create model
    print("\nStep 2: Creating GCN model...")
    num_features = data.num_node_features
    model = TrafficGCN(num_node_features=num_features)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of classes: 3 (Low/Medium/High traffic)")
    print(f"  Device: {DEVICE}")
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    trainer = GCNTrainer(model, device=DEVICE, learning_rate=LEARNING_RATE)
    
    history = trainer.train(
        data=data,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_path=MODEL_SAVE_PATH
    )
    
    # Step 4: Evaluate on test set
    print("\nStep 4: Evaluating on test set...")
    test_loss, test_acc = trainer.evaluate(data, data.test_mask)
    
    # Calculate detailed metrics
    metrics = calculate_metrics(data, model, data.test_mask)
    class_names = ['Low Traffic', 'Medium Traffic', 'High Traffic']
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Step 5: Generate visualizations and reports
    print("\nStep 5: Generating visualizations and evaluation reports...")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(data, model, data.test_mask, class_names)
    
    # Print evaluation summary
    print_evaluation_summary(metrics, class_names)
    
    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"✓ Best model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Test accuracy: {test_acc:.4f}")
    print(f"✓ Training history plot: models/gcn_training_history.png")
    print(f"✓ Confusion matrix: models/gcn_confusion_matrix.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

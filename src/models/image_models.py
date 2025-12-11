"""
3 Image Classification Models for Traffic Incident Severity
Models: ResNet-18, MobileNet-V2, EfficientNet-B0
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

NUM_CLASSES = 4
CLASS_NAMES = ['none', 'minor', 'moderate', 'severe']

# Severity multipliers for pathfinding
SEVERITY_MULTIPLIERS = {
    'none': 1.0,
    'minor': 1.2,
    'moderate': 1.5,
    'severe': 2.0
}

# Model 1: ResNet-18
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

# Model 2: MobileNet-V2
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

# Model 3: EfficientNet-B0
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

# Loading functions
def load_resnet18(path: str, device: str = 'cuda'):
    model = ResNet18Model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model

def load_mobilenet(path: str, device: str = 'cuda'):
    model = MobileNetV2Model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model

def load_efficientnet(path: str, device: str = 'cuda'):
    model = EfficientNetModel(num_classes=NUM_CLASSES)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model

def predict_severity(model, image_tensor, device='cuda'):
    """Predict severity from image"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(output, dim=1).item()
    
    severity = CLASS_NAMES[pred_class]
    confidence = probabilities[pred_class].item()
    
    return severity, confidence

def get_edge_multiplier(severity: str) -> float:
    """Get edge weight multiplier based on severity"""
    return SEVERITY_MULTIPLIERS.get(severity, 1.0)

def ensemble_predict(models: list, image_tensor, device='cuda'):
    """Ensemble prediction using multiple models (majority voting)"""
    predictions = []
    confidences = []
    
    for model in models:
        severity, conf = predict_severity(model, image_tensor, device)
        predictions.append(severity)
        confidences.append(conf)
    
    # Majority vote
    from collections import Counter
    vote = Counter(predictions).most_common(1)[0][0]
    avg_conf = sum(confidences) / len(confidences)
    
    return vote, avg_conf
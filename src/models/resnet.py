import torch
import torch.nn as nn
from typing import Optional, Dict, Any

def get_model_resnet34(pretrained: bool = True, num_classes: Optional[int] = None, **kwargs: Dict[str, Any]) -> nn.Module:
    """Create and return a ResNet34 model instance.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int, optional): Number of output classes. If None, keeps the default
        **kwargs: Additional arguments to pass to the ResNet constructor
    
    Returns:
        nn.Module: Configured ResNet34 model instance
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
    
    if num_classes is not None:
        # head modification
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    return model
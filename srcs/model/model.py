import torch
import torchvision


def get_mobilenet_model(num_classes: int = 29):
    model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.mobilenet.MobileNet_V3_Small_Weights)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def get_efficientnet_model(num_classes: int = 29):
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

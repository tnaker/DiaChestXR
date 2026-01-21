import torch
import torch.nn as nn
from torchvision import models

def get_covid_alexnet(num_classes = 3, pretrained = True):
    model = models.alexnet(weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
    for param in model.features.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model

# Single Label Model, Softmax Cross Entropy Loss

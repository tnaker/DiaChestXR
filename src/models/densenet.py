import torch 
import torch.nn as nn
from torchvision import models

def get_lession_densenet(num_classes = 18, pretrained = True):
    model = models.densenet121(weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

# Multi Label Model, Sigmoid Cross Entropy Loss
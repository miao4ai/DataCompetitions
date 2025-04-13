import torch
import torch.nn as nn
import timm

class CLEFModel(nn.Module):
    def __init__(self, model_name="tf_efficientnetv2_s", num_classes=182):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, num_classes)

    def forward(self, x):
        return self.model(x)

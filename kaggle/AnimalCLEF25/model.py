import timm
import torch.nn as nn

def get_model(model_name: str, num_classes: int):
    model = timm.create_model(model_name, pretrained=True)

    # 自动替换分类头
    if hasattr(model, 'classifier'):  # EfficientNetV2
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):  # ConvNeXt, ViT, Swin
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):  # ResNet fallback
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown classifier head for model {model_name}")
    
    return model
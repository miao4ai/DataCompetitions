import timm
import torch.nn as nn

import timm
import torch.nn as nn

MODEL_REGISTRY = {
    'convnext_base': lambda num_classes: timm.create_model('convnext_base', pretrained=True, num_classes=num_classes),
    'convnext_large': lambda num_classes: timm.create_model('convnext_large', pretrained=True, num_classes=num_classes),
    'swin_base_patch4_window7_224': lambda num_classes: timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes),
    'vit_base_patch16_224': lambda num_classes: timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes),
    
    # 💡 新增的多样性模型
    'vit_base_patch32_224': lambda num_classes: timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=num_classes),
    'vit_base_patch16_224_in21k': lambda num_classes: timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=num_classes),
    'beit_base_patch16_224': lambda num_classes: timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=num_classes),
    'swinv2_base_window12to16_192to256_22kft1k': lambda num_classes: timm.create_model('swinv2_base_window12to16_192to256_22kft1k', pretrained=True, num_classes=num_classes),
    'coatnet_0': lambda num_classes: timm.create_model('coatnet_0', pretrained=True, num_classes=num_classes),
    'coatnet_1': lambda num_classes: timm.create_model('coatnet_1', pretrained=True, num_classes=num_classes),
}


def get_model(model_name: str, num_classes: int):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not supported.")
    return MODEL_REGISTRY[model_name](num_classes)

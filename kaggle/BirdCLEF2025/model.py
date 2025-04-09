from torch import nn
import timm

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,  
            in_chans=cfg.in_channels,
            drop_rate=0.0,    
            drop_path_rate=0.0
        )
        
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits
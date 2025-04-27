import torch
import torch.nn as nn
import timm
from torchvision import transforms

class CatDogClassifier(nn.Module):
    def __init__(self, num_labels=2, backbone_name='convnextv2_base.fcmae_ft_in22k_in1k', pretrained_backbone=True):
        super().__init__()
        
        # Use ConvNeXtV2 as the backbone for image feature extraction
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,  # No classification head
        )
        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, 512)

        # Binary Classification head (dog or cat)
        self.classifier = nn.Linear(512, 1) 

    def forward(self, images):
        b, c, h, w = images.shape
        # Extract features from backbone
        feats = self.backbone(images).view(b, -1)
        feats = self.proj(feats)  # (b, 512)
        
        # Classify using the classifier
        logits = self.classifier(feats)  # (b, 1)
        return logits
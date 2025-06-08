import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation.lraspp import LRASPP_MobileNet_V3_Large_Weights


class FoodSegLRASPP(nn.Module):
    """
    LRASPP model for food segmentation.
    """
    
    def __init__(self, num_classes=104, pretrained=True):
        super(FoodSegLRASPP, self).__init__()
        
        # Load pretrained LRASPP model
        if pretrained:
            weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = lraspp_mobilenet_v3_large(weights=weights)
        else:
            self.model = lraspp_mobilenet_v3_large(weights=None)
        
        # Modify the classifier for our number of classes
        # The original model has 21 classes (COCO + VOC), we need 104
        in_channels = self.model.classifier.low_classifier.in_channels
        
        # Replace the classifier
        self.model.classifier = LRASPPHead(
            in_channels=in_channels,
            low_channels=40,  # MobileNetV3-Large low-level feature channels
            num_classes=num_classes,
            inter_channels=128
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.model(x)


class LRASPPHead(nn.Module):
    """
    Custom LRASPP Head for food segmentation.
    """
    
    def __init__(self, in_channels, low_channels, num_classes, inter_channels=128):
        super(LRASPPHead, self).__init__()
        
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)
    
    def forward(self, input):
        low = input["low"]
        high = input["high"]
        
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        
        return self.low_classifier(low) + self.high_classifier(x)


def create_model(num_classes=104, pretrained=True):
    """
    Create LRASPP model for food segmentation.
    
    Args:
        num_classes (int): Number of segmentation classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        FoodSegLRASPP: The segmentation model
    """
    return FoodSegLRASPP(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model):
    """Count the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = create_model(num_classes=104)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output['out'].shape}")

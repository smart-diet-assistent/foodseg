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
    # Import necessary modules for dataset testing
    import matplotlib.pyplot as plt
    import numpy as np
    from dataset import prepare_dataset, FoodSegDataset, get_val_transforms
    from config import NUM_CLASSES
    
    print("="*60)
    print("Testing LRASPP Model with FoodSeg Test Dataset")
    print("="*60)
    
    # Test the model creation
    model = create_model(num_classes=NUM_CLASSES)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Load dataset
    print("\nLoading FoodSeg103 dataset...")
    try:
        dataset, label_mapping = prepare_dataset()
        
        # Check which splits are available
        available_splits = list(dataset.keys())
        print(f"Available dataset splits: {available_splits}")
        
        # Use test split if available, otherwise use validation or train
        test_split = None
        if 'test' in dataset:
            test_split = dataset['test']
            split_name = 'test'
        elif 'validation' in dataset:
            test_split = dataset['validation']
            split_name = 'validation'
        else:
            test_split = dataset['train']
            split_name = 'train'
        
        print(f"Using '{split_name}' split with {len(test_split)} samples")
        
        # Create dataset instance
        test_dataset = FoodSegDataset(
            test_split, 
            transform=get_val_transforms(),
            label_mapping=label_mapping
        )
        
        print(f"Test dataset created with {len(test_dataset)} samples")
        
        # Test with a single sample from the dataset
        print(f"\nTesting with sample from {split_name} dataset...")
        
        # Get first sample
        sample_idx = 0
        image, mask = test_dataset[sample_idx]
        
        print(f"Sample {sample_idx}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Unique mask values: {torch.unique(mask).tolist()}")
        
        # Add batch dimension and test forward pass
        x = image.unsqueeze(0)  # Add batch dimension
        print(f"\nTesting forward pass...")
        print(f"Input batch shape: {x.shape}")
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            output = model(x)
            print(f"Output shape: {output['out'].shape}")
            print(f"Output dtype: {output['out'].dtype}")
            
            # Get predictions
            predictions = torch.argmax(output['out'], dim=1)
            print(f"Prediction shape: {predictions.shape}")
            print(f"Unique prediction values: {torch.unique(predictions).tolist()}")
        
        print(f"\n✅ Model test completed successfully!")
        print(f"✅ Model can process real FoodSeg103 {split_name} data")
        
        # Additional info
        if label_mapping:
            print(f"\nLabel mapping info:")
            print(f"  Original → Mapped labels: {len(label_mapping)} classes")
            print(f"  Mapping: {dict(list(label_mapping.items())[:5])}..." if len(label_mapping) > 5 else f"  Mapping: {label_mapping}")
        
    except Exception as e:
        print(f"❌ Error testing with dataset: {str(e)}")
        print("Falling back to synthetic data test...")
        
        # Fallback to synthetic data
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(x)
            print(f"Synthetic test - Input shape: {x.shape}")
            print(f"Synthetic test - Output shape: {output['out'].shape}")
    
    print("="*60)

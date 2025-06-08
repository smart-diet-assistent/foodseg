import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2


def pixel_accuracy(pred, target, ignore_index=255):
    """
    Calculate pixel accuracy.
    
    Args:
        pred (torch.Tensor): Predicted segmentation masks
        target (torch.Tensor): Ground truth masks
        ignore_index (int): Index to ignore in calculation
    
    Returns:
        float: Pixel accuracy
    """
    valid = (target != ignore_index)
    acc_sum = (pred[valid] == target[valid]).sum()
    valid_sum = valid.sum()
    
    if valid_sum == 0:
        return 0.0
    
    return float(acc_sum) / float(valid_sum)


def mean_iou(pred, target, num_classes, ignore_index=255):
    """
    Calculate mean Intersection over Union (mIoU).
    
    Args:
        pred (torch.Tensor): Predicted segmentation masks
        target (torch.Tensor): Ground truth masks
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in calculation
    
    Returns:
        float: Mean IoU
        list: Per-class IoU scores
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    # Remove NaN values for mean calculation
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    
    if len(valid_ious) == 0:
        return 0.0, ious
    
    return np.mean(valid_ious), ious


def dice_score(pred, target, num_classes, ignore_index=255, smooth=1e-6):
    """
    Calculate Dice score (F1 score for segmentation).
    
    Args:
        pred (torch.Tensor): Predicted segmentation masks
        target (torch.Tensor): Ground truth masks
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in calculation
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Mean Dice score
        list: Per-class Dice scores
    """
    dice_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        pred_inds = (pred == cls).float()
        target_inds = (target == cls).float()
        
        intersection = (pred_inds * target_inds).sum()
        total = pred_inds.sum() + target_inds.sum()
        
        if total == 0:
            dice_scores.append(float('nan'))
        else:
            dice = (2.0 * intersection + smooth) / (total + smooth)
            dice_scores.append(dice.item())
    
    # Remove NaN values for mean calculation
    valid_scores = [score for score in dice_scores if not np.isnan(score)]
    
    if len(valid_scores) == 0:
        return 0.0, dice_scores
    
    return np.mean(valid_scores), dice_scores


class SegmentationLoss(nn.Module):
    """
    Combined loss for semantic segmentation.
    """
    
    def __init__(self, ce_weight=1.0, dice_weight=0.5, ignore_index=255):
        super(SegmentationLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Calculate Dice loss."""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Ignore index handling
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        total = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (total + smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss, ce_loss, dice_loss


def visualize_predictions(images, predictions, targets, class_names=None, num_samples=4):
    """
    Visualize segmentation predictions.
    
    Args:
        images (torch.Tensor): Input images
        predictions (torch.Tensor): Predicted masks
        targets (torch.Tensor): Ground truth masks
        class_names (list): List of class names
        num_samples (int): Number of samples to visualize
    """
    # Move to CPU and convert to numpy
    images = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Show original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Show prediction
        pred_colored = create_colored_mask(predictions[i])
        axes[i, 1].imshow(pred_colored)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        
        # Show ground truth
        target_colored = create_colored_mask(targets[i])
        axes[i, 2].imshow(target_colored)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig


def create_colored_mask(mask, num_classes=104):
    """
    Create a colored mask for visualization.
    
    Args:
        mask (np.ndarray): Segmentation mask
        num_classes (int): Number of classes
    
    Returns:
        np.ndarray: Colored mask
    """
    # Create a color palette
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20))])
    colors = np.vstack([colors, plt.cm.tab20c(np.linspace(0, 1, 20))])
    
    # Repeat colors if we have more classes
    while len(colors) < num_classes:
        colors = np.vstack([colors, colors])
    
    colors = colors[:num_classes]
    
    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3))
    for class_id in range(num_classes):
        mask_class = (mask == class_id)
        colored_mask[mask_class] = colors[class_id][:3]
    
    return colored_mask


def calculate_metrics(predictions, targets, num_classes):
    """
    Calculate all segmentation metrics.
    
    Args:
        predictions (torch.Tensor): Predicted masks
        targets (torch.Tensor): Ground truth masks
        num_classes (int): Number of classes
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Convert to numpy for easier calculation
    pred_np = predictions.cpu().numpy().flatten()
    target_np = targets.cpu().numpy().flatten()
    
    # Calculate metrics
    pixel_acc = pixel_accuracy(predictions, targets)
    miou, class_ious = mean_iou(predictions, targets, num_classes)
    dice, class_dice = dice_score(predictions, targets, num_classes)
    
    metrics = {
        'pixel_accuracy': pixel_acc,
        'mean_iou': miou,
        'class_ious': class_ious,
        'dice_score': dice,
        'class_dice': class_dice
    }
    
    return metrics


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from tqdm import tqdm


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


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    """
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255, class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, 
                                 weight=self.class_weights, 
                                 ignore_index=self.ignore_index, 
                                 reduction='none')
        
        # Calculate focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class SegmentationLoss(nn.Module):
    """
    Enhanced loss for semantic segmentation with focal loss and class weights.
    """
    
    def __init__(self, ce_weight=1.0, dice_weight=0.5, focal_weight=0.5, 
                 ignore_index=255, class_weights=None, use_focal=True,
                 focal_alpha=1.0, focal_gamma=2.0):
        super(SegmentationLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index
        self.use_focal = use_focal
        
        # Standard CrossEntropy loss with class weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        
        # Focal loss for hard examples
        if use_focal:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha, 
                gamma=focal_gamma, 
                ignore_index=ignore_index,
                class_weights=class_weights
            )
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Calculate Dice loss with class balancing."""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Ignore index handling
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        # Calculate Dice loss per class
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        total = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (total + smooth)
        
        # Weight classes by inverse frequency (if class weights available)
        if hasattr(self, 'ce_loss') and self.ce_loss.weight is not None:
            class_weights = self.ce_loss.weight.to(dice.device)
            # Normalize weights
            normalized_weights = class_weights / class_weights.sum() * len(class_weights)
            dice = dice * normalized_weights.unsqueeze(0)
        
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def forward(self, pred, target):
        # Standard Cross Entropy Loss
        ce_loss = self.ce_loss(pred, target)
        
        # Dice Loss
        dice_loss = self.dice_loss(pred, target)
        
        # Focal Loss (if enabled)
        focal_loss = 0.0
        if self.use_focal:
            focal_loss = self.focal_loss(pred, target)
        
        # Combined loss
        total_loss = (self.ce_weight * ce_loss + 
                     self.dice_weight * dice_loss + 
                     self.focal_weight * focal_loss)
        
        return total_loss, ce_loss, dice_loss, focal_loss


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


def calculate_class_weights(dataset, num_classes, method='inverse_freq', smooth=1.0):
    """
    Calculate class weights for addressing class imbalance.
    
    Args:
        dataset: Dataset to analyze
        num_classes: Number of classes
        method: Weight calculation method ('inverse_freq', 'median_freq', 'effective_num')
        smooth: Smoothing factor
    
    Returns:
        torch.Tensor: Class weights
    """
    print("Calculating class weights from dataset...")
    
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.float32)
    total_pixels = 0
    
    # Count pixels for each class
    for i, (_, mask) in enumerate(tqdm(dataset, desc="Analyzing class distribution")):
        if i % 100 == 0 and i > 0:  # Sample every 100th to speed up for large datasets
            continue
            
        unique_classes, counts = torch.unique(mask, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            if cls < num_classes:
                class_pixel_counts[cls] += count.float()
                total_pixels += count.item()
    
    # Calculate weights based on method
    if method == 'inverse_freq':
        # Inverse frequency weighting
        class_frequencies = class_pixel_counts / total_pixels
        class_weights = 1.0 / (class_frequencies + smooth)
        
    elif method == 'median_freq':
        # Median frequency weighting
        median_freq = torch.median(class_pixel_counts[class_pixel_counts > 0])
        class_weights = median_freq / (class_pixel_counts + smooth)
        
    elif method == 'effective_num':
        # Effective number of samples weighting
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_pixel_counts)
        class_weights = (1.0 - beta) / (effective_num + 1e-8)
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights so that they sum to num_classes
    class_weights = class_weights / class_weights.mean()
    
    # Set background weight to lower value to reduce its dominance
    if class_weights[0] > 0:
        class_weights[0] = class_weights[0] * 0.5
    
    print(f"Class weights calculated using {method}:")
    for i, weight in enumerate(class_weights):
        if class_pixel_counts[i] > 0:
            print(f"  Class {i}: weight={weight:.4f}, pixels={int(class_pixel_counts[i])}")
    
    return class_weights


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with warmup and restarts.
    """
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=0.1, min_lr=0.001, 
                 warmup_steps=0, gamma=1., last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr 
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * \
                    (1 + np.cos(np.pi * (self.step_in_cycle - self.warmup_steps) / 
                               (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(np.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = np.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

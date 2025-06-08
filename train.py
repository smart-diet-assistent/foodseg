import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import wandb
import tempfile

from dataset import prepare_dataset, create_dataloaders
from model import create_model
from utils import SegmentationLoss, calculate_metrics, visualize_predictions, EarlyStopping
from config import *
from wandb_config import get_wandb_config


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{total_epochs}")
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        predictions = outputs['out']
        
        # Calculate loss
        total_loss_batch, ce_loss, dice_loss = criterion(predictions, masks)
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += total_loss_batch.item()
        total_ce_loss += ce_loss.item()
        total_dice_loss += dice_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'CE': f'{ce_loss.item():.4f}',
            'Dice': f'{dice_loss.item():.4f}'
        })
        
        # Log batch metrics to wandb (configurable frequency)
        wandb_cfg = get_wandb_config()
        if batch_idx % wandb_cfg["batch_log_frequency"] == 0:
            wandb.log({
                "batch/train_loss": total_loss_batch.item(),
                "batch/train_ce_loss": ce_loss.item(),
                "batch/train_dice_loss": dice_loss.item(),
                "batch/learning_rate": optimizer.param_groups[0]['lr'],
                "batch/epoch": epoch + (batch_idx / len(train_loader))
            })
    
    avg_loss = total_loss / len(train_loader)
    avg_ce_loss = total_ce_loss / len(train_loader)
    avg_dice_loss = total_dice_loss / len(train_loader)
    
    return avg_loss, avg_ce_loss, avg_dice_loss


def validate_epoch(model, val_loader, criterion, device, num_classes, epoch, total_epochs):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0
    
    all_predictions = []
    all_targets = []
    sample_images = []
    
    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{total_epochs}")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Store first batch for visualization
            if batch_idx == 0:
                sample_images = images[:4].cpu()
            
            # Forward pass
            outputs = model(images)
            predictions = outputs['out']
            
            # Calculate loss
            total_loss_batch, ce_loss, dice_loss = criterion(predictions, masks)
            
            # Update losses
            total_loss += total_loss_batch.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()
            
            # Get predictions for metrics
            pred_masks = torch.argmax(predictions, dim=1)
            all_predictions.append(pred_masks.cpu())
            all_targets.append(masks.cpu())
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Dice': f'{dice_loss.item():.4f}'
            })
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = calculate_metrics(all_predictions, all_targets, num_classes)
    
    avg_loss = total_loss / len(val_loader)
    avg_ce_loss = total_ce_loss / len(val_loader)
    avg_dice_loss = total_dice_loss / len(val_loader)
    
    return avg_loss, avg_ce_loss, avg_dice_loss, metrics, all_predictions[:4], all_targets[:4], sample_images


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def train():
    """Main training function."""
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset, label_mapping = prepare_dataset()
    train_loader, val_loader = create_dataloaders(dataset, label_mapping, BATCH_SIZE, NUM_WORKERS)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    # 如果有标签映射，使用映射后的类别数，否则使用配置中的默认值
    num_classes = len(label_mapping) if label_mapping else NUM_CLASSES
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Number of classes: {num_classes}")
    
    # Loss and optimizer
    criterion = SegmentationLoss(ce_weight=1.0, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='max')
    
    # Initialize Wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"foodseg_lraspp_{timestamp}"
    
    # Get wandb configuration
    wandb_cfg = get_wandb_config()
    
    # Wandb training configuration
    wandb_config = {
        # Model configuration
        "model_name": MODEL_NAME,
        "backbone": BACKBONE,
        "num_classes": num_classes,
        "total_parameters": total_params,
        
        # Dataset configuration
        "dataset_name": DATASET_NAME,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "val_split": VAL_SPLIT,
        "num_workers": NUM_WORKERS,
        
        # Label filtering
        "desired_labels": DESIRED_LABELS,
        "remap_labels": REMAP_LABELS,
        "min_label_pixels": MIN_LABEL_PIXELS,
        "label_mapping": label_mapping,
        
        # Training configuration
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        
        # Loss configuration
        "loss_function": "CE + Dice",
        "ce_weight": 1.0,
        "dice_weight": 0.5,
        
        # Data augmentation
        "augmentation_prob": AUGMENTATION_PROB,
        "rotation_limit": ROTATION_LIMIT,
        "brightness_limit": BRIGHTNESS_LIMIT,
        
        # System configuration
        "device": str(device),
        "pin_memory": PIN_MEMORY,
        
        # Preprocessing
        "normalization_mean": MEAN,
        "normalization_std": STD
    }
    
    # Initialize wandb run
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=run_name,
        config=wandb_config,
        mode=wandb_cfg["mode"],
        tags=wandb_cfg["tags"],
        notes=wandb_cfg["notes"] or f"Training LRASPP model on FoodSeg103 with {num_classes} classes"
    )
    
    # Watch model gradients and parameters if enabled
    if wandb_cfg["watch_model"]:
        log_setting = wandb_cfg["watch_log"] if wandb_cfg["log_gradients"] or wandb_cfg["log_parameters"] else None
        if log_setting:
            wandb.watch(model, log=log_setting, log_freq=wandb_cfg["watch_log_freq"])
    
    # Tensorboard writer (keeping for compatibility)
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    
    # Training loop
    best_miou = 0.0
    training_history = {
        'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_dice': [], 'val_pixel_acc': []
    }
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"Wandb run: {wandb.run.url}")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Training
        train_loss, train_ce_loss, train_dice_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )
        
        # Validation
        val_loss, val_ce_loss, val_dice_loss, metrics, sample_preds, sample_targets, sample_images = validate_epoch(
            model, val_loader, criterion, device, num_classes, epoch, NUM_EPOCHS
        )
        
        # Update learning rate
        scheduler.step(metrics['mean_iou'])
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}, Dice: {train_dice_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (CE: {val_ce_loss:.4f}, Dice: {val_dice_loss:.4f})")
        print(f"Val mIoU: {metrics['mean_iou']:.4f}")
        print(f"Val Dice: {metrics['dice_score']:.4f}")
        print(f"Val Pixel Acc: {metrics['pixel_accuracy']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Wandb logging
        wandb_metrics = {
            # Training metrics
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/ce_loss": train_ce_loss,
            "train/dice_loss": train_dice_loss,
            
            # Validation metrics
            "val/loss": val_loss,
            "val/ce_loss": val_ce_loss,
            "val/dice_loss": val_dice_loss,
            "val/mean_iou": metrics['mean_iou'],
            "val/dice_score": metrics['dice_score'],
            "val/pixel_accuracy": metrics['pixel_accuracy'],
            
            # Learning rate
            "learning_rate": current_lr,
            
            # Per-class IoU (if available)
            **{f"val/iou_class_{i}": iou for i, iou in enumerate(metrics.get('per_class_iou', []))},
            
            # Additional metrics
            "best_miou": best_miou
        }
        
        # Create prediction visualization for wandb
        if len(sample_images) > 0 and wandb_cfg["save_prediction_artifacts"]:
            # Create visualization figure
            fig = visualize_predictions(sample_images, sample_preds, sample_targets)
            
            # Save to temporary file and log to wandb
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                fig.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                wandb_metrics["predictions"] = wandb.Image(
                    tmp_file.name,
                    caption=f"Epoch {epoch+1} - Predictions vs Ground Truth"
                )
                os.unlink(tmp_file.name)
            plt.close(fig)
        
        # Log all metrics to wandb
        wandb.log(wandb_metrics)
        
        # Tensorboard logging (keeping for compatibility)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metrics/mIoU', metrics['mean_iou'], epoch)
        writer.add_scalar('Metrics/Dice', metrics['dice_score'], epoch)
        writer.add_scalar('Metrics/PixelAcc', metrics['pixel_accuracy'], epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Update history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_miou'].append(metrics['mean_iou'])
        training_history['val_dice'].append(metrics['dice_score'])
        training_history['val_pixel_acc'].append(metrics['pixel_accuracy'])
        
        # Save best model
        if metrics['mean_iou'] > best_miou:
            best_miou = metrics['mean_iou']
            best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, metrics, best_model_path)
            
            # Log best model as artifact to wandb
            artifact = wandb.Artifact(
                name=f"best_model_{run_name}",
                type="model",
                description=f"Best model checkpoint at epoch {epoch+1} with mIoU {best_miou:.4f}"
            )
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
            
            # Save and log visualization of best predictions
            fig = visualize_predictions(sample_images, sample_preds, sample_targets)
            best_pred_path = os.path.join(RESULTS_DIR, f'best_predictions_epoch_{epoch+1}.png')
            plt.savefig(best_pred_path, dpi=300, bbox_inches='tight')
            
            # Log best predictions as artifact
            pred_artifact = wandb.Artifact(
                name=f"best_predictions_{run_name}",
                type="predictions",
                description=f"Best model predictions at epoch {epoch+1}"
            )
            pred_artifact.add_file(best_pred_path)
            wandb.log_artifact(pred_artifact)
            plt.close(fig)
            
            # Update wandb summary with best metrics
            wandb.run.summary["best_epoch"] = epoch + 1
            wandb.run.summary["best_miou"] = best_miou
            wandb.run.summary["best_dice"] = metrics['dice_score']
            wandb.run.summary["best_pixel_acc"] = metrics['pixel_accuracy']
        
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path)
            
            # Log checkpoint as artifact
            checkpoint_artifact = wandb.Artifact(
                name=f"checkpoint_epoch_{epoch+1}_{run_name}",
                type="checkpoint",
                description=f"Model checkpoint at epoch {epoch+1}"
            )
            checkpoint_artifact.add_file(checkpoint_path)
            wandb.log_artifact(checkpoint_artifact)
        
        # Early stopping check
        if early_stopping(metrics['mean_iou']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            wandb.run.summary["early_stopped"] = True
            wandb.run.summary["early_stop_epoch"] = epoch + 1
            break
    
    # Final checkpoint
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'final_model.pth')
    save_checkpoint(model, optimizer, epoch, metrics, final_model_path)
    
    # Log final model as artifact
    final_artifact = wandb.Artifact(
        name=f"final_model_{run_name}",
        type="model",
        description=f"Final model checkpoint after {epoch+1} epochs"
    )
    final_artifact.add_file(final_model_path)
    wandb.log_artifact(final_artifact)
    
    # Save training history
    history_path = os.path.join(RESULTS_DIR, 'training_history.npz')
    np.savez(history_path, **training_history)
    
    # Log training history as artifact
    history_artifact = wandb.Artifact(
        name=f"training_history_{run_name}",
        type="results",
        description="Complete training history and metrics"
    )
    history_artifact.add_file(history_path)
    wandb.log_artifact(history_artifact)
    
    # Plot and log training curves
    plot_training_curves(training_history, run_name)
    
    # Log training curves to wandb
    curves_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    wandb.log({"training_curves": wandb.Image(curves_path, caption="Training Progress Curves")})
    
    # Final summary
    wandb.run.summary["total_epochs"] = epoch + 1
    wandb.run.summary["final_miou"] = metrics['mean_iou']
    wandb.run.summary["final_dice"] = metrics['dice_score']
    wandb.run.summary["final_pixel_acc"] = metrics['pixel_accuracy']
    wandb.run.summary["total_parameters"] = total_params
    
    writer.close()
    wandb.finish()
    
    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")
    print(f"Wandb run completed: {wandb.run.url}")


def plot_training_curves(history, run_name=None):
    """Plot training curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # mIoU curve
    axes[0, 1].plot(epochs, history['val_miou'], label='Val mIoU', color='green')
    axes[0, 1].set_title('Mean IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice score curve
    axes[1, 0].plot(epochs, history['val_dice'], label='Val Dice', color='orange')
    axes[1, 0].set_title('Dice Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Pixel accuracy curve
    axes[1, 1].plot(epochs, history['val_pixel_acc'], label='Val Pixel Acc', color='purple')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Pixel Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Add overall title with run name if provided
    if run_name:
        fig.suptitle(f'Training Progress - {run_name}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    train()

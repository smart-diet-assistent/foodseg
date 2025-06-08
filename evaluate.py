import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import prepare_dataset, create_dataloaders, get_val_transforms, FoodSegDataset
from model import create_model
from utils import calculate_metrics, visualize_predictions
from config import *


def evaluate_model(model_path=None, dataset_split='test'):
    """
    Evaluate the trained model on test set.
    
    Args:
        model_path (str): Path to the saved model
        dataset_split (str): Dataset split to evaluate on ('test' or 'validation')
    """
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = prepare_dataset()
    
    # Create test dataloader
    if dataset_split in dataset:
        test_dataset = FoodSegDataset(dataset[dataset_split], transform=get_val_transforms())
    else:
        print(f"Split '{dataset_split}' not found, using validation split")
        test_dataset = FoodSegDataset(dataset['validation'], transform=get_val_transforms())
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load model
    print("Loading model...")
    model = create_model(NUM_CLASSES, pretrained=False)
    
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {model_path}")
        print(f"Model was trained for {checkpoint['epoch']} epochs")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Evaluation
    print("Starting evaluation...")
    all_predictions = []
    all_targets = []
    sample_images = []
    sample_predictions = []
    sample_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = outputs['out']
            pred_masks = torch.argmax(predictions, dim=1)
            
            # Store predictions and targets
            all_predictions.append(pred_masks.cpu())
            all_targets.append(masks.cpu())
            
            # Store samples for visualization (first batch only)
            if batch_idx == 0:
                sample_images = images[:8].cpu()  # First 8 images
                sample_predictions = pred_masks[:8].cpu()
                sample_targets = masks[:8].cpu()
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"Total predictions: {all_predictions.shape}")
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(all_predictions, all_targets, NUM_CLASSES)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Dice Score: {metrics['dice_score']:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class IoU (showing top 10 classes):")
    class_ious = metrics['class_ious']
    valid_ious = [(i, iou) for i, iou in enumerate(class_ious) if not np.isnan(iou)]
    valid_ious.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_idx, iou) in enumerate(valid_ious[:10]):
        print(f"Class {class_idx}: {iou:.4f}")
    
    print(f"\nPer-class Dice Score (showing top 10 classes):")
    class_dice = metrics['class_dice']
    valid_dice = [(i, dice) for i, dice in enumerate(class_dice) if not np.isnan(dice)]
    valid_dice.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_idx, dice) in enumerate(valid_dice[:10]):
        print(f"Class {class_idx}: {dice:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Visualize sample predictions
    fig = visualize_predictions(
        sample_images, 
        sample_predictions, 
        sample_targets, 
        num_samples=8
    )
    plt.savefig(os.path.join(RESULTS_DIR, 'evaluation_predictions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot metrics distribution
    plot_metrics_distribution(metrics)
    
    # Save metrics to file
    results_file = os.path.join(RESULTS_DIR, 'evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n\n")
        f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
        f.write(f"Dice Score: {metrics['dice_score']:.4f}\n\n")
        
        f.write(f"Per-class IoU:\n")
        for i, iou in enumerate(class_ious):
            if not np.isnan(iou):
                f.write(f"Class {i}: {iou:.4f}\n")
        
        f.write(f"\nPer-class Dice Score:\n")
        for i, dice in enumerate(class_dice):
            if not np.isnan(dice):
                f.write(f"Class {i}: {dice:.4f}\n")
    
    print(f"Results saved to: {results_file}")
    print("Evaluation completed!")
    
    return metrics


def plot_metrics_distribution(metrics):
    """Plot distribution of per-class metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # IoU distribution
    class_ious = metrics['class_ious']
    valid_ious = [iou for iou in class_ious if not np.isnan(iou)]
    
    axes[0].hist(valid_ious, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Distribution of Per-Class IoU')
    axes[0].set_xlabel('IoU Score')
    axes[0].set_ylabel('Number of Classes')
    axes[0].axvline(np.mean(valid_ious), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(valid_ious):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice distribution
    class_dice = metrics['class_dice']
    valid_dice = [dice for dice in class_dice if not np.isnan(dice)]
    
    axes[1].hist(valid_dice, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Distribution of Per-Class Dice Score')
    axes[1].set_xlabel('Dice Score')
    axes[1].set_ylabel('Number of Classes')
    axes[1].axvline(np.mean(valid_dice), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(valid_dice):.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def compare_models(model_paths, model_names=None):
    """
    Compare multiple models on the test set.
    
    Args:
        model_paths (list): List of paths to model checkpoints
        model_names (list): List of model names for display
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    results = []
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model_path)
        results.append({
            'name': model_name,
            'pixel_accuracy': metrics['pixel_accuracy'],
            'mean_iou': metrics['mean_iou'],
            'dice_score': metrics['dice_score']
        })
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results))
    width = 0.25
    
    pixel_acc = [r['pixel_accuracy'] for r in results]
    mean_iou = [r['mean_iou'] for r in results]
    dice_score = [r['dice_score'] for r in results]
    
    ax.bar(x - width, pixel_acc, width, label='Pixel Accuracy', alpha=0.8)
    ax.bar(x, mean_iou, width, label='Mean IoU', alpha=0.8)
    ax.bar(x + width, dice_score, width, label='Dice Score', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<20} {'Pixel Acc':<12} {'Mean IoU':<12} {'Dice Score':<12}")
    print("-"*70)
    
    for result in results:
        print(f"{result['name']:<20} {result['pixel_accuracy']:<12.4f} "
              f"{result['mean_iou']:<12.4f} {result['dice_score']:<12.4f}")


if __name__ == "__main__":
    # Evaluate the best model
    evaluate_model()
    
    # Example: Compare multiple models
    # model_paths = [
    #     "models/best_model.pth",
    #     "models/checkpoint_epoch_30.pth",
    #     "models/final_model.pth"
    # ]
    # model_names = ["Best Model", "Epoch 30", "Final Model"]
    # compare_models(model_paths, model_names)

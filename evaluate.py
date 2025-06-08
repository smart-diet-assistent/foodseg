import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset import prepare_dataset, create_dataloaders, get_val_transforms, FoodSegDataset
from model import create_model
from utils import calculate_metrics, visualize_predictions
from config import *


def evaluate_model(model_path=None, dataset_split='validation'):
    """
    Evaluate the trained model on validation set with comprehensive metrics.
    
    Args:
        model_path (str): Path to the saved model
        dataset_split (str): Dataset split to evaluate on ('validation' or 'test')
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
    dataset, label_mapping = prepare_dataset()
    
    # Determine number of classes
    num_classes = len(label_mapping) if label_mapping else NUM_CLASSES
    print(f"Number of classes: {num_classes}")
    
    # Create validation dataloader
    if dataset_split in dataset:
        eval_dataset = FoodSegDataset(dataset[dataset_split], 
                                     transform=get_val_transforms(),
                                     label_mapping=label_mapping)
    else:
        print(f"Split '{dataset_split}' not found, using validation split")
        eval_dataset = FoodSegDataset(dataset['validation'], 
                                     transform=get_val_transforms(),
                                     label_mapping=label_mapping)
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Evaluation batches: {len(eval_loader)}")
    
    # Load model
    print("Loading model...")
    model = create_model(num_classes, pretrained=False)
    
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {model_path}")
        print(f"Model was trained for {checkpoint['epoch']} epochs")
        if 'metrics' in checkpoint:
            print(f"Best training metrics: {checkpoint['metrics']}")
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
    
    # Clear GPU memory before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(eval_loader, desc="Evaluating")):
            try:
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                masks = masks.to(device, dtype=torch.long, non_blocking=True)
                
                # Validate mask values
                max_class_value = masks.max().item()
                if max_class_value >= num_classes:
                    print(f"Warning: Found class {max_class_value} but model has {num_classes} classes")
                    masks = torch.clamp(masks, 0, num_classes - 1)
                
                # Forward pass
                outputs = model(images)
                predictions = outputs['out']
                pred_masks = torch.argmax(predictions, dim=1)
                
                # Apply 1% area threshold: filter out predictions with area < 1%
                batch_size, height, width = pred_masks.shape
                total_pixels = height * width
                area_threshold = 0.02  # 1%
                min_pixels = int(total_pixels * area_threshold)
                
                # Filter small predictions for each image in batch
                for i in range(batch_size):
                    mask = pred_masks[i]
                    unique_classes = torch.unique(mask)
                    
                    for class_id in unique_classes:
                        if class_id == 0:  # Skip background
                            continue
                        class_pixels = (mask == class_id).sum().item()
                        if class_pixels < min_pixels:
                            # Set small regions to background (class 0)
                            pred_masks[i][mask == class_id] = 0
                
                # Store predictions and targets
                all_predictions.append(pred_masks.cpu())
                all_targets.append(masks.cpu())
                
                # Store samples for visualization (first batch only)
                if batch_idx == 0:
                    sample_images = images[:8].cpu()  # First 8 images
                    sample_predictions = pred_masks[:8].cpu()
                    sample_targets = masks[:8].cpu()
                    
                # Periodically clear memory
                if batch_idx % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU memory error at batch {batch_idx}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"Total predictions: {all_predictions.shape}")
    
    # Calculate comprehensive metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(all_predictions, all_targets, num_classes)
    
    # Print detailed results
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*70)
    
    # Basic metrics
    print(f"Basic Segmentation Metrics:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Dice Score: {metrics['dice_score']:.4f}")
    
    # Image-level Precision and Recall metrics
    print(f"\nImage-Level Precision & Recall (Class Presence Based):")
    print(f"  Image Macro Precision: {metrics['image_macro_precision']:.4f}")
    print(f"  Image Macro Recall: {metrics['image_macro_recall']:.4f}")
    print(f"  Image Macro F1-Score: {metrics['image_macro_f1']:.4f}")
    print(f"  Class Macro Precision: {metrics['image_class_macro_precision']:.4f}")
    print(f"  Class Macro Recall: {metrics['image_class_macro_recall']:.4f}")
    print(f"  Class Macro F1-Score: {metrics['image_class_macro_f1']:.4f}")
    
    # Class analysis summary
    class_analysis = metrics['class_analysis']
    print(f"\nImage-Level Class Analysis:")
    print(f"  Total Images: {class_analysis['total_images']}")
    print(f"  Perfect Matches: {class_analysis['perfect_matches']} ({class_analysis['perfect_matches']/class_analysis['total_images']*100:.1f}%)")
    print(f"  Partial Matches: {class_analysis['partial_matches']} ({class_analysis['partial_matches']/class_analysis['total_images']*100:.1f}%)")
    print(f"  No Matches: {class_analysis['no_matches']} ({class_analysis['no_matches']/class_analysis['total_images']*100:.1f}%)")
    
    # Show some detailed examples
    if class_analysis['detailed_results']:
        print(f"\nSample Image Predictions (first 5):")
        for i, result in enumerate(class_analysis['detailed_results'][:5]):
            print(f"  Image {result['image_index']}: Pred={result['predicted_classes']}, True={result['target_classes']}, "
                  f"Jaccard={result['jaccard_index']:.3f}")
    
    # Per-class metrics (top performing classes)
    print(f"\nTop 10 Classes by IoU:")
    class_ious = metrics['class_ious']
    valid_ious = [(i, iou) for i, iou in enumerate(class_ious) if not np.isnan(iou)]
    valid_ious.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_idx, iou) in enumerate(valid_ious[:10]):
        image_precision = metrics['image_class_precisions'][class_idx]
        image_recall = metrics['image_class_recalls'][class_idx]
        image_f1 = metrics['image_class_f1_scores'][class_idx]
        print(f"Class {class_idx:2d}: IoU={iou:.4f} | Image P/R/F1={image_precision:.3f}/{image_recall:.3f}/{image_f1:.3f}")
    
    print(f"\nTop 10 Classes by Image-Level Precision:")
    image_precisions = metrics['image_class_precisions']
    valid_image_precisions = [(i, p) for i, p in enumerate(image_precisions) if not np.isnan(p)]
    valid_image_precisions.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_idx, precision) in enumerate(valid_image_precisions[:10]):
        iou = metrics['class_ious'][class_idx]
        recall = metrics['image_class_recalls'][class_idx]
        f1 = metrics['image_class_f1_scores'][class_idx]
        print(f"Class {class_idx:2d}: Image Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, IoU={iou:.4f}")
    
    print(f"\nTop 10 Classes by Image-Level Recall:")
    image_recalls = metrics['image_class_recalls']
    valid_image_recalls = [(i, r) for i, r in enumerate(image_recalls) if not np.isnan(r)]
    valid_image_recalls.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_idx, recall) in enumerate(valid_image_recalls[:10]):
        iou = metrics['class_ious'][class_idx]
        precision = metrics['image_class_precisions'][class_idx]
        f1 = metrics['image_class_f1_scores'][class_idx]
        print(f"Class {class_idx:2d}: Image Rec={recall:.4f}, Prec={precision:.4f}, F1={f1:.4f}, IoU={iou:.4f}")
    
    # Create comprehensive visualizations
    print("\nCreating visualizations...")
    # Create validation results subdirectory
    val_results_dir = os.path.join(RESULTS_DIR, 'val')
    os.makedirs(val_results_dir, exist_ok=True)
    
    # Visualize sample predictions
    if len(sample_images) > 0:
        print("  Generating prediction visualizations...")
        fig = visualize_predictions(
            sample_images, 
            sample_predictions, 
            sample_targets, 
            num_samples=min(8, len(sample_images))
        )
        plt.savefig(os.path.join(val_results_dir, 'evaluation_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Plot comprehensive metrics distribution
    print("  Creating comprehensive metrics plots...")
    plot_comprehensive_metrics(metrics, num_classes, val_results_dir)
    
    # Create confusion matrix visualization for a subset of classes
    print("  Generating confusion matrix...")
    create_confusion_matrix_viz(all_predictions, all_targets, num_classes, val_results_dir)
    
    # Save comprehensive metrics to file
    print("  Saving detailed results...")
    save_detailed_results(metrics, model_path, dataset_split, num_classes, label_mapping, val_results_dir)
    
    print("Evaluation completed!")
    return metrics


def plot_comprehensive_metrics(metrics, num_classes, output_dir=None):
    """Plot comprehensive distribution of all metrics."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # IoU distribution
    class_ious = metrics['class_ious']
    valid_ious = [iou for iou in class_ious if not np.isnan(iou)]
    
    axes[0, 0].hist(valid_ious, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Per-Class IoU')
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].set_ylabel('Number of Classes')
    axes[0, 0].axvline(np.mean(valid_ious), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_ious):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice distribution
    class_dice = metrics['class_dice']
    valid_dice = [dice for dice in class_dice if not np.isnan(dice)]
    
    axes[0, 1].hist(valid_dice, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Distribution of Per-Class Dice Score')
    axes[0, 1].set_xlabel('Dice Score')
    axes[0, 1].set_ylabel('Number of Classes')
    axes[0, 1].axvline(np.mean(valid_dice), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_dice):.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Image-Level Precision distribution
    image_class_precisions = metrics['image_class_precisions']
    valid_precisions = [p for p in image_class_precisions if not np.isnan(p)]
    
    axes[0, 2].hist(valid_precisions, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_title('Distribution of Image-Level Per-Class Precision')
    axes[0, 2].set_xlabel('Precision')
    axes[0, 2].set_ylabel('Number of Classes')
    axes[0, 2].axvline(np.mean(valid_precisions), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_precisions):.3f}')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Image-Level Recall distribution
    image_class_recalls = metrics['image_class_recalls']
    valid_recalls = [r for r in image_class_recalls if not np.isnan(r)]
    
    axes[1, 0].hist(valid_recalls, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_title('Distribution of Image-Level Per-Class Recall')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Number of Classes')
    axes[1, 0].axvline(np.mean(valid_recalls), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_recalls):.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Image-Level F1 Score distribution
    image_class_f1s = metrics['image_class_f1_scores']
    valid_f1s = [f1 for f1 in image_class_f1s if not np.isnan(f1)]
    
    axes[1, 1].hist(valid_f1s, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_title('Distribution of Image-Level Per-Class F1-Score')
    axes[1, 1].set_xlabel('F1-Score')
    axes[1, 1].set_ylabel('Number of Classes')
    axes[1, 1].axvline(np.mean(valid_f1s), color='darkred', linestyle='--', 
                       label=f'Mean: {np.mean(valid_f1s):.3f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary metrics comparison
    metrics_names = ['Pixel Acc', 'Mean IoU', 'Dice', 'Image Macro Prec', 'Image Macro Rec', 'Image Macro F1']
    metrics_values = [
        metrics['pixel_accuracy'],
        metrics['mean_iou'],
        metrics['dice_score'],
        metrics['image_macro_precision'],
        metrics['image_macro_recall'],
        metrics['image_macro_f1']
    ]
    
    bars = axes[1, 2].bar(metrics_names, metrics_values, alpha=0.7, 
                          color=['blue', 'green', 'orange', 'purple', 'red', 'brown'])
    axes[1, 2].set_title('Overall Metrics Summary')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_confusion_matrix_viz(predictions, targets, num_classes, output_dir=None, max_classes=20):
    """Create confusion matrix visualization for top classes."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    # Convert to flat arrays
    pred_flat = predictions.numpy().flatten()
    target_flat = targets.numpy().flatten()
    
    # Find most frequent classes
    unique_classes, counts = np.unique(target_flat, return_counts=True)
    top_classes_idx = np.argsort(counts)[-max_classes:]
    top_classes = unique_classes[top_classes_idx]
    
    # Filter predictions and targets to include only top classes
    mask = np.isin(target_flat, top_classes)
    pred_filtered = pred_flat[mask]
    target_filtered = target_flat[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(target_filtered, pred_filtered, labels=top_classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=top_classes, yticklabels=top_classes)
    plt.title(f'Normalized Confusion Matrix (Top {len(top_classes)} Classes)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results(metrics, model_path, dataset_split, num_classes, label_mapping=None, output_dir=None):
    """Save comprehensive evaluation results to files."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    # Save text results
    results_file = os.path.join(output_dir, 'comprehensive_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Comprehensive Evaluation Results\n")
        f.write(f"===============================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset Split: {dataset_split}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")
        
        # Overall metrics
        f.write(f"Overall Metrics:\n")
        f.write(f"----------------\n")
        f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
        f.write(f"Dice Score: {metrics['dice_score']:.4f}\n\n")
        
        f.write(f"Precision & Recall Metrics:\n")
        f.write(f"---------------------------\n")
        f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
        f.write(f"Micro Precision: {metrics['micro_precision']:.4f}\n")
        f.write(f"Micro Recall: {metrics['micro_recall']:.4f}\n")
        f.write(f"Micro F1-Score: {metrics['micro_f1']:.4f}\n\n")
        
        # Per-class detailed metrics
        f.write(f"Per-Class Detailed Metrics:\n")
        f.write(f"---------------------------\n")
        f.write(f"{'Class':<8} {'IoU':<8} {'Dice':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}\n")
        f.write(f"-" * 56 + "\n")
        
        for i in range(num_classes):
            iou = metrics['class_ious'][i]
            dice = metrics['class_dice'][i]
            prec = metrics['class_precisions'][i]
            recall = metrics['class_recalls'][i]
            f1 = metrics['class_f1_scores'][i]
            
            # Only write classes that have valid metrics
            if not (np.isnan(iou) and np.isnan(dice) and np.isnan(prec)):
                f.write(f"{i:<8} {iou:<8.4f} {dice:<8.4f} {prec:<8.4f} {recall:<8.4f} {f1:<8.4f}\n")
    
    # Save metrics as numpy arrays for later analysis
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.npz')
    np.savez(metrics_file, **metrics)
    
    # Save label mapping if available
    if label_mapping:
        import json
        label_file = os.path.join(output_dir, 'label_mapping_used.json')
        with open(label_file, 'w') as f:
            json.dump(label_mapping, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    print(f"Metrics arrays saved to: {metrics_file}")


def plot_metrics_distribution(metrics):
    """Plot distribution of per-class metrics (legacy function for compatibility)."""
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


def compare_models(model_paths, model_names=None, dataset_split='validation'):
    """
    Compare multiple models on the validation set.
    
    Args:
        model_paths (list): List of paths to model checkpoints
        model_names (list): List of model names for display
        dataset_split (str): Dataset split to use for comparison
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    results = []
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model_path, dataset_split)
        if metrics:
            results.append({
                'name': model_name,
                'pixel_accuracy': metrics['pixel_accuracy'],
                'mean_iou': metrics['mean_iou'],
                'dice_score': metrics['dice_score'],
                'image_macro_precision': metrics['image_macro_precision'],
                'image_macro_recall': metrics['image_macro_recall'],
                'image_macro_f1': metrics['image_macro_f1']
            })
    
    if not results:
        print("No valid results to compare!")
        return
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(results))
    width = 0.35
    
    # Plot 1: Basic metrics
    pixel_acc = [r['pixel_accuracy'] for r in results]
    mean_iou = [r['mean_iou'] for r in results]
    dice_score = [r['dice_score'] for r in results]
    
    axes[0, 0].bar(x - width/2, pixel_acc, width/3, label='Pixel Accuracy', alpha=0.8)
    axes[0, 0].bar(x, mean_iou, width/3, label='Mean IoU', alpha=0.8)
    axes[0, 0].bar(x + width/2, dice_score, width/3, label='Dice Score', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Basic Segmentation Metrics')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([r['name'] for r in results], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Image-Level Precision and Recall
    image_macro_prec = [r['image_macro_precision'] for r in results]
    image_macro_rec = [r['image_macro_recall'] for r in results]
    
    axes[0, 1].bar(x - width/2, image_macro_prec, width, label='Image Macro Precision', alpha=0.8)
    axes[0, 1].bar(x + width/2, image_macro_rec, width, label='Image Macro Recall', alpha=0.8)
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Image-Level Precision & Recall')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([r['name'] for r in results], rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Scores
    image_macro_f1 = [r['image_macro_f1'] for r in results]
    
    axes[1, 0].bar(x, image_macro_f1, width, label='Image Macro F1', alpha=0.8)
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Scores')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([r['name'] for r in results], rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Overall ranking
    # Calculate overall score as weighted average
    overall_scores = []
    for r in results:
        score = (r['mean_iou'] * 0.3 + r['dice_score'] * 0.3 + 
                r['image_macro_f1'] * 0.2 + r['pixel_accuracy'] * 0.2)
        overall_scores.append(score)
    
    bars = axes[1, 1].bar(x, overall_scores, alpha=0.8, color='purple')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Overall Score')
    axes[1, 1].set_title('Overall Performance Ranking')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([r['name'] for r in results], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, overall_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comprehensive_model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Print comprehensive comparison table
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*100)
    print(f"{'Model':<15} {'Pixel Acc':<10} {'Mean IoU':<10} {'Dice':<10} {'Macro P':<10} {'Macro R':<10} {'Macro F1':<10} {'Micro F1':<10} {'Overall':<10}")
    print("-"*100)
    
    for i, result in enumerate(results):
        overall_score = overall_scores[i]
        print(f"{result['name']:<15} {result['pixel_accuracy']:<10.4f} "
              f"{result['mean_iou']:<10.4f} {result['dice_score']:<10.4f} "
              f"{result['macro_precision']:<10.4f} {result['macro_recall']:<10.4f} "
              f"{result['macro_f1']:<10.4f} {result['micro_f1']:<10.4f} {overall_score:<10.4f}")
    
    # Save comparison results
    comparison_file = os.path.join(RESULTS_DIR, 'model_comparison_results.txt')
    with open(comparison_file, 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("========================\n\n")
        f.write(f"Dataset Split: {dataset_split}\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Model: {result['name']}\n")
            f.write(f"  Pixel Accuracy: {result['pixel_accuracy']:.4f}\n")
            f.write(f"  Mean IoU: {result['mean_iou']:.4f}\n")
            f.write(f"  Dice Score: {result['dice_score']:.4f}\n")
            f.write(f"  Macro Precision: {result['macro_precision']:.4f}\n")
            f.write(f"  Macro Recall: {result['macro_recall']:.4f}\n")
            f.write(f"  Macro F1: {result['macro_f1']:.4f}\n")
            f.write(f"  Micro F1: {result['micro_f1']:.4f}\n")
            f.write(f"  Overall Score: {overall_scores[i]:.4f}\n\n")
    
    print(f"Comparison results saved to: {comparison_file}")
    return results


if __name__ == "__main__":
    # Evaluate the best model on validation dataset
    print("Evaluating best model on validation dataset...")
    metrics = evaluate_model(dataset_split='validation')
    
    # Print summary
    if metrics:
        print(f"\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        print(f"Overall Performance:")
        print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  Dice Score: {metrics['dice_score']:.4f}")
        print(f"  Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"  Micro F1-Score: {metrics['micro_f1']:.4f}")
    
    # Example: Compare multiple models
    # print("\nComparing multiple model checkpoints...")
    # model_paths = [
    #     "models/best_model.pth",
    #     "models/checkpoint_epoch_30.pth",
    #     "models/final_model.pth"
    # ]
    # model_names = ["Best Model", "Epoch 30", "Final Model"]
    # compare_models(model_paths, model_names)

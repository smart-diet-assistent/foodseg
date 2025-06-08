import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import create_model
from utils import create_colored_mask
from config import *


def load_model(model_path):
    """Load trained model."""
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Create model
    model = create_model(NUM_CLASSES, pretrained=False)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return None, None
    
    model = model.to(device)
    model.eval()
    
    return model, device


def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Preprocess image for inference."""
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    original_shape = image.shape[:2]
    
    # Define preprocessing pipeline
    transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    
    # Apply transforms
    transformed = transform(image=image)
    processed_image = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return processed_image, original_shape, image


def postprocess_prediction(prediction, original_shape):
    """Postprocess prediction to original image size."""
    # Get prediction mask
    pred_mask = torch.argmax(prediction, dim=1).squeeze(0)
    
    # Convert to numpy
    pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
    
    # Resize to original image size
    pred_mask_resized = cv2.resize(
        pred_mask, 
        (original_shape[1], original_shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    return pred_mask_resized


def predict_single_image(model, device, image_path, save_path=None, show_result=True):
    """
    Predict segmentation for a single image.
    
    Args:
        model: Trained segmentation model
        device: Device to run inference on
        image_path (str): Path to input image
        save_path (str): Path to save result (optional)
        show_result (bool): Whether to display the result
    
    Returns:
        tuple: (original_image, prediction_mask, colored_mask)
    """
    # Preprocess image
    processed_image, original_shape, original_image = preprocess_image(image_path)
    processed_image = processed_image.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(processed_image)
        prediction = outputs['out']
    
    # Postprocess prediction
    pred_mask = postprocess_prediction(prediction, original_shape)
    
    # Create colored visualization
    colored_mask = create_colored_mask(pred_mask, NUM_CLASSES)
    
    # Create overlay
    overlay = cv2.addWeighted(original_image.astype(np.float32), 0.6, 
                             (colored_mask * 255).astype(np.float32), 0.4, 0)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Visualize results
    if show_result:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_mask, cmap='tab20')
        axes[1].set_title('Prediction Mask')
        axes[1].axis('off')
        
        axes[2].imshow(colored_mask)
        axes[2].set_title('Colored Mask')
        axes[2].axis('off')
        
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Result saved to: {save_path}")
        
        plt.show()
    
    return original_image, pred_mask, colored_mask, overlay


def predict_batch_images(model, device, image_dir, output_dir):
    """
    Predict segmentation for a batch of images.
    
    Args:
        model: Trained segmentation model
        device: Device to run inference on
        image_dir (str): Directory containing input images
        output_dir (str): Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) 
                           if f.lower().endswith(ext.lower())])
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")
        
        image_path = os.path.join(image_dir, image_file)
        
        # Create output paths
        base_name = os.path.splitext(image_file)[0]
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        colored_path = os.path.join(output_dir, f"{base_name}_colored.png")
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        result_path = os.path.join(output_dir, f"{base_name}_result.png")
        
        try:
            # Run prediction
            original, pred_mask, colored_mask, overlay = predict_single_image(
                model, device, image_path, show_result=False
            )
            
            # Save results
            cv2.imwrite(mask_path, pred_mask)
            cv2.imwrite(colored_path, (colored_mask * 255).astype(np.uint8))
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Save combined result
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(original)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(pred_mask, cmap='tab20')
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            
            axes[2].imshow(colored_mask)
            axes[2].set_title('Colored Mask')
            axes[2].axis('off')
            
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(result_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    print(f"Batch inference completed! Results saved to: {output_dir}")


def get_prediction_confidence(model, device, image_path):
    """
    Get prediction confidence scores.
    
    Args:
        model: Trained segmentation model
        device: Device to run inference on
        image_path (str): Path to input image
    
    Returns:
        tuple: (prediction_mask, confidence_map, mean_confidence)
    """
    # Preprocess image
    processed_image, original_shape, _ = preprocess_image(image_path)
    processed_image = processed_image.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(processed_image)
        prediction = outputs['out']
        
        # Get probabilities
        probabilities = F.softmax(prediction, dim=1)
        
        # Get max probabilities (confidence)
        max_probs, pred_classes = torch.max(probabilities, dim=1)
    
    # Postprocess
    pred_mask = postprocess_prediction(prediction, original_shape)
    
    # Resize confidence map
    confidence_map = max_probs.squeeze(0).cpu().numpy()
    confidence_map = cv2.resize(confidence_map, (original_shape[1], original_shape[0]))
    
    mean_confidence = float(confidence_map.mean())
    
    return pred_mask, confidence_map, mean_confidence


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Food Segmentation Inference')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(MODEL_SAVE_DIR, 'best_model.pth'),
                       help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Output directory for results')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in directory')
    parser.add_argument('--confidence', action='store_true',
                       help='Show prediction confidence')
    
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model_path)
    if model is None:
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.batch and os.path.isdir(args.image_path):
        # Batch processing
        predict_batch_images(model, device, args.image_path, args.output_dir)
    
    elif os.path.isfile(args.image_path):
        # Single image processing
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        save_path = os.path.join(args.output_dir, f"{base_name}_result.png")
        
        original, pred_mask, colored_mask, overlay = predict_single_image(
            model, device, args.image_path, save_path=save_path
        )
        
        if args.confidence:
            pred_mask_conf, confidence_map, mean_conf = get_prediction_confidence(
                model, device, args.image_path
            )
            
            print(f"Mean confidence: {mean_conf:.4f}")
            
            # Visualize confidence
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(confidence_map, cmap='viridis')
            plt.title(f'Confidence Map (Mean: {mean_conf:.3f})')
            plt.colorbar()
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Prediction Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            conf_save_path = os.path.join(args.output_dir, f"{base_name}_confidence.png")
            plt.savefig(conf_save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    else:
        print(f"Invalid path: {args.image_path}")


if __name__ == "__main__":
    main()

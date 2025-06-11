# FoodSeg103 Semantic Segmentation with LRASPP

> English | [简体中文](README_zh.md)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project implements semantic segmentation on the FoodSeg103 dataset using the LRASPP (Lite Reduced Atrous Spatial Pyramid Pooling) model, with advanced data filtering capabilities.

## Dataset
- **FoodSeg103**: A large-scale food image segmentation dataset with 103 food categories
- **Source**: EduardoPacheco/FoodSeg103 on Hugging Face
- **Features**: Support for label filtering and class subset selection

## Model
- **LRASPP**: Lite Reduced Atrous Spatial Pyramid Pooling
- **Backbone**: MobileNetV3-Large
- **Purpose**: Lightweight semantic segmentation for mobile applications

## Key Features

### 🎯 Label Filtering System
- **Selective Training**: Choose specific food categories to train on
- **Label Remapping**: Automatically remap labels to continuous IDs (0, 1, 2, ...)
- **Pixel Threshold**: Filter out samples with insufficient label pixels
- **Automatic Analysis**: Built-in dataset analysis with recommendations

### 📊 Dataset Analysis
- Comprehensive label distribution analysis
- Automatic filtering recommendations
- Visual statistics and histograms
- Export analysis results to JSON

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Analyze the dataset (optional but recommended):
```bash
python analyze_dataset.py
```

3. Configure label filtering in `config.py`:
```python
# Example: Train on first 20 food categories only
DESIRED_LABELS = list(range(0, 21))  # 0=background, 1-20=food classes
REMAP_LABELS = True
MIN_LABEL_PIXELS = 100
```

4. Download and prepare the dataset:
```bash
python prepare_dataset.py
```

5. Train the model:
```bash
python train.py
```

6. Evaluate the model:
```bash
python evaluate.py
```

7. Run inference:
```bash
python inference.py --image_path path/to/image.jpg
```

## Label Filtering Configuration

### Configuration Options

```python
# In config.py

# Option 1: Train on specific labels
DESIRED_LABELS = [0, 1, 2, 5, 10, 15, 20]  # Background + 6 food classes

# Option 2: Train on first N classes
DESIRED_LABELS = list(range(0, 21))  # Background + 20 food classes

# Option 3: Use all classes (no filtering)
DESIRED_LABELS = None

# Label remapping (recommended: True)
REMAP_LABELS = True

# Minimum pixels per label in sample
MIN_LABEL_PIXELS = 100
```

### Recommended Workflows

1. **Quick Start (Small Subset)**:
   ```python
   DESIRED_LABELS = [0, 1, 2, 3, 4, 5]  # 6 classes
   REMAP_LABELS = True
   MIN_LABEL_PIXELS = 100
   ```

2. **Balanced Training (Medium Subset)**:
   ```python
   DESIRED_LABELS = list(range(0, 21))  # 21 classes
   REMAP_LABELS = True
   MIN_LABEL_PIXELS = 100
   ```

3. **Full Dataset**:
   ```python
   DESIRED_LABELS = None  # All 104 classes
   REMAP_LABELS = False
   MIN_LABEL_PIXELS = 50
   ```

## Project Structure
```
foodseg/
├── requirements.txt              # Dependencies
├── config.py                    # Configuration with filtering options
├── dataset.py                   # Dataset loading with filtering
├── model.py                     # LRASPP model implementation
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Inference script
├── utils.py                     # Utility functions
├── prepare_dataset.py           # Dataset preparation
├── analyze_dataset.py           # Dataset analysis tool
├── label_filtering_examples.py  # Configuration examples
├── README.md                    # English documentation
├── README_zh.md                 # Chinese documentation
├── LICENSE                      # MIT License
├── wandb_config.py              # Weights & Biases configuration
├── cache_manager.py             # Cache management utilities
├── CACHE_README.md              # Cache system documentation
├── convert_to_tflite.py         # TensorFlow Lite conversion
├── test_inference.py            # Inference testing
├── test_lraspp_inputs.py        # Model input testing
└── inference_demo.py            # Interactive inference demo

# Generated directories:
├── data/                        # Dataset cache
│   └── label_mapping.json      # Label mapping file
├── cache/                       # Hugging Face dataset cache
│   └── filtered_datasets/      # Filtered dataset cache
├── models/                      # Saved models
│   ├── best_model.pth          # PyTorch model
│   ├── food_segmentation_model.onnx  # ONNX format
│   ├── food_segmentation_model.tflite # TensorFlow Lite
│   └── food_segmentation_model_saved_model/ # TensorFlow SavedModel
├── results/                     # Training results and visualizations
├── logs/                        # Training logs
├── analysis_results/            # Dataset analysis results
├── inference_output/            # Inference output images
├── esp32_deployment/            # ESP32 deployment files
└── FoodSegmentationLibrary/     # Arduino library for ESP32
```

## Model Export & Deployment

### ONNX Export
```bash
# Convert trained model to ONNX format
python convert_to_onnx.py --model_path models/best_model.pth
```

### TensorFlow Lite Export
```bash
# Convert to TensorFlow Lite for mobile deployment
python convert_to_tflite.py --model_path models/best_model.pth
```

### ESP32 Deployment
The project includes ESP32 deployment support:
- Converted model weights in `esp32_deployment/model_data.cc`
- Arduino library in `FoodSegmentationLibrary/`
- Ready for microcontroller deployment

### Interactive Demo
```bash
# Run interactive inference demo
python inference_demo.py
```

## Advanced Usage

### Dataset Analysis
```bash
# Analyze label distribution and get recommendations
python analyze_dataset.py
```
This will generate:
- `analysis_results/dataset_analysis.json`: Detailed statistics
- `analysis_results/recommended_configs.json`: Filtering recommendations  
- `analysis_results/*_distribution.png`: Visualization plots

### Custom Label Selection
```python
# Example: Focus on common food items
DESIRED_LABELS = [
    0,   # background
    1,   # apple
    5,   # banana
    10,  # bread
    15,  # pizza
    20,  # hamburger
    # ... add more as needed
]
```

### Batch Inference
```bash
# Process all images in a directory
python inference.py --image_path ./test_images/ --batch --output_dir ./results/
```

## Benefits of Label Filtering

1. **🚀 Faster Training**: Reduced dataset size means faster iterations
2. **💾 Memory Efficiency**: Lower memory requirements for smaller class sets
3. **🎯 Focused Learning**: Better performance on selected categories
4. **⚖️ Balanced Classes**: Avoid class imbalance issues
5. **🔧 Easy Debugging**: Simpler to analyze and debug with fewer classes

## Training & Evaluation

### Training Results
After training, you'll find various outputs in the `results/` directory:
- `best_predictions_epoch_*.png`: Visualization of best predictions per epoch
- `comprehensive_evaluation_results.txt`: Detailed evaluation metrics
- `comprehensive_metrics.png`: Performance visualization
- `confusion_matrix.png`: Class confusion matrix
- `evaluation_metrics.npz`: Numerical evaluation data

### Monitoring Training
The project supports Weights & Biases integration:
```bash
# Configure W&B (optional)
python wandb_config.py

# Training with W&B logging
python train.py --use_wandb
```

### Cache Management
Efficient dataset caching system:
```bash
# Check cache status
python cache_manager.py --status

# Clear specific cache
python cache_manager.py --clear filtered

# See CACHE_README.md for detailed cache management
```

## Testing

### Model Testing
```bash
# Test LRASPP model inputs
python test_lraspp_inputs.py

# Test inference pipeline
python test_inference.py --image_path image.jpg
```

## Performance Tips

1. **Start Small**: Begin with 5-10 classes for initial experiments
2. **Use Analysis**: Run `analyze_dataset.py` to understand your data
3. **Enable Remapping**: Always set `REMAP_LABELS=True` for filtered datasets
4. **Adjust Thresholds**: Tune `MIN_LABEL_PIXELS` based on your target object sizes
5. **Monitor Metrics**: Check class-wise IoU to identify problematic classes

## Troubleshooting

### Common Issues
- **Empty dataset after filtering**: Check if `DESIRED_LABELS` exist in the dataset
- **Memory errors**: Reduce batch size or number of classes
- **Poor performance**: Ensure sufficient samples per class (>100 recommended)

### Debug Commands
```bash
# Test dataset loading
python dataset.py

# Check model architecture
python model.py

# Validate configuration
python -c "from config import *; print(f'Classes: {NUM_CLASSES}, Labels: {DESIRED_LABELS}')"
```

## FAQ

### Q: How do I choose the right number of classes for my use case?
A: Start with 5-10 classes for experimentation, then gradually increase. Use `analyze_dataset.py` to understand class distribution and balance.

### Q: What's the difference between REMAP_LABELS=True and False?
A: When True, labels are remapped to continuous IDs (0,1,2,3...). When False, original label IDs are preserved. Always use True for filtered datasets.

### Q: My training is very slow. How can I speed it up?
A: 
- Reduce the number of classes with label filtering
- Use a smaller batch size if memory is limited
- Enable mixed precision training
- Use filtered datasets to reduce data loading time

### Q: Can I use this project for other segmentation datasets?
A: Yes, modify the dataset loading in `dataset.py` to support your data format. The LRASPP model can work with any semantic segmentation task.

### Q: How do I deploy the model to mobile devices?
A: Use the TensorFlow Lite conversion script `convert_to_tflite.py` to create a mobile-optimized model, then integrate it into your mobile app.

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM for full dataset
- 4GB+ RAM for filtered datasets

### Dependencies
```bash
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
datasets>=2.0.0
Pillow>=8.0.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0

# Optional dependencies
onnx>=1.12.0           # For ONNX export
tensorflow>=2.8.0      # For TensorFlow Lite export
wandb>=0.12.0          # For experiment tracking
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Issues and pull requests are welcome! Please ensure you follow the project's coding standards.

## Acknowledgments

- The creators of the FoodSeg103 dataset
- PyTorch team for the LRASPP implementation
- Hugging Face for dataset hosting services

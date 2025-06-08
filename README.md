# FoodSeg103 Semantic Segmentation with LRASPP

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
└── README.md                    # This file

# Generated directories:
├── data/                        # Dataset cache
│   └── label_mapping.json      # Label mapping file
├── models/                      # Saved models
├── results/                     # Training results
├── logs/                        # Training logs
└── analysis_results/            # Dataset analysis results
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

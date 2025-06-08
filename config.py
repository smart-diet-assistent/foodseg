import os
import torch

# Dataset configuration
DATASET_NAME = "EduardoPacheco/FoodSeg103"
DATA_DIR = "./data"
CACHE_DIR = "./cache"

# Model configuration
MODEL_NAME = "lraspp_mobilenet_v3_large"
NUM_CLASSES = 104  # 103 food classes + background
BACKBONE = "mobilenet_v3_large"

# Label filtering configuration
# 指定需要保留的标签ID列表，None表示保留所有标签
DESIRED_LABELS = [0, 1, 2, 3, 4, 5]  # 测试用：背景+5个食物类别
# 是否重新映射标签到连续的ID (0, 1, 2, ...)
REMAP_LABELS = True
# 最小像素阈值，标签在mask中的像素数少于此值的样本将被过滤
MIN_LABEL_PIXELS = 100

# Training configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Image configuration
IMAGE_SIZE = (512, 512)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training parameters
DEVICE = "cuda" if os.path.exists("/proc/driver/nvidia") else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
NUM_WORKERS = 4
PIN_MEMORY = True

# Validation configuration
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# Output configuration
MODEL_SAVE_DIR = "./models"
RESULTS_DIR = "./results"
LOG_DIR = "./logs"

# Augmentation parameters
AUGMENTATION_PROB = 0.5
ROTATION_LIMIT = 15
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2

# Evaluation metrics
METRICS = ["pixel_accuracy", "mean_iou", "class_iou", "dice_score"]

import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from config import *


def filter_dataset_by_labels(dataset_split, desired_labels=None, min_label_pixels=100):
    """
    筛选数据集，只保留包含指定标签的样本。
    
    Args:
        dataset_split: 数据集分割（train/validation/test）
        desired_labels: 需要保留的标签ID列表，None表示保留所有标签
        min_label_pixels: 最小像素阈值，标签像素数少于此值的样本将被过滤
    
    Returns:
        filtered_indices: 筛选后的有效样本索引列表
        label_mapping: 标签重新映射字典（原标签ID -> 新标签ID）
    """
    print(f"开始筛选数据集...")
    print(f"原始样本数: {len(dataset_split)}")
    
    if desired_labels is None:
        print("未指定标签筛选，保留所有样本")
        return list(range(len(dataset_split))), None
    
    print(f"指定保留的标签: {desired_labels}")
    print(f"最小像素阈值: {min_label_pixels}")
    
    filtered_indices = []
    label_stats = {}
    
    # 遍历所有样本
    for idx in tqdm(range(len(dataset_split)), desc="筛选样本"):
        try:
            sample = dataset_split[idx]
            
            # 获取mask
            mask = sample['label'] if 'label' in sample else sample['mask']
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            
            # 确保mask是单通道
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            # 获取mask中的唯一标签
            unique_labels = np.unique(mask)
            
            # 检查是否包含指定标签
            has_desired_label = False
            valid_pixels = 0
            
            for label in unique_labels:
                if label in desired_labels:
                    label_pixels = np.sum(mask == label)
                    if label_pixels >= min_label_pixels:
                        has_desired_label = True
                        valid_pixels += label_pixels
                        
                        # 统计标签出现次数
                        if label not in label_stats:
                            label_stats[label] = 0
                        label_stats[label] += 1
            
            # 如果包含所需标签且满足像素阈值，则保留该样本
            if has_desired_label:
                filtered_indices.append(idx)
                
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            continue
    
    print(f"筛选后样本数: {len(filtered_indices)}")
    print(f"筛选比例: {len(filtered_indices)/len(dataset_split)*100:.2f}%")
    
    # 打印标签统计信息
    print("\n标签统计:")
    for label, count in sorted(label_stats.items()):
        print(f"标签 {label}: {count} 个样本")
    
    # 创建标签重新映射
    label_mapping = None
    if REMAP_LABELS and desired_labels:
        # 只为实际出现的标签创建映射
        present_labels = sorted(list(label_stats.keys()))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(present_labels)}
        print(f"\n标签映射: {label_mapping}")
    
    return filtered_indices, label_mapping


def prepare_dataset():
    """Download and prepare the FoodSeg103 dataset."""
    print("Loading FoodSeg103 dataset...")
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    
    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"Validation samples: {len(dataset['validation'])}")
    if 'test' in dataset:
        print(f"Test samples: {len(dataset['test'])}")
    
    # 数据筛选
    filtered_dataset = {}
    global_label_mapping = None
    
    for split_name in dataset.keys():
        print(f"\n筛选 {split_name} 数据集...")
        
        # 筛选当前分割的数据
        filtered_indices, label_mapping = filter_dataset_by_labels(
            dataset[split_name], 
            desired_labels=DESIRED_LABELS,
            min_label_pixels=MIN_LABEL_PIXELS
        )
        
        # 保存标签映射（使用训练集的映射作为全局映射）
        if split_name == 'train' and label_mapping is not None:
            global_label_mapping = label_mapping
        
        # 创建筛选后的数据集
        if filtered_indices:
            filtered_split = dataset[split_name].select(filtered_indices)
            filtered_dataset[split_name] = filtered_split
            print(f"{split_name} 筛选完成: {len(filtered_split)} 个样本")
        else:
            print(f"警告: {split_name} 筛选后没有有效样本")
    
    # 保存标签映射到全局变量或文件
    if global_label_mapping is not None:
        import json
        mapping_file = os.path.join(DATA_DIR, 'label_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(global_label_mapping, f, indent=2)
        print(f"标签映射已保存到: {mapping_file}")
        
        # 更新类别数量
        global NUM_CLASSES
        NUM_CLASSES = len(global_label_mapping)
        print(f"更新后的类别数量: {NUM_CLASSES}")
    
    return filtered_dataset, global_label_mapping


class FoodSegDataset(Dataset):
    """
    Custom dataset for FoodSeg103 semantic segmentation.
    """
    
    def __init__(self, dataset_split, transform=None, target_size=IMAGE_SIZE, label_mapping=None):
        self.dataset = dataset_split
        self.transform = transform
        self.target_size = target_size
        self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get image and mask
        image = sample['image']
        mask = sample['label'] if 'label' in sample else sample['mask']
        
        # Convert to numpy arrays
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel if RGB
        
        # Apply label mapping if provided
        if self.label_mapping is not None:
            mapped_mask = np.zeros_like(mask)
            for old_label, new_label in self.label_mapping.items():
                mapped_mask[mask == old_label] = new_label
            # 将未映射的标签设为背景（0）
            unmapped_pixels = np.ones_like(mask, dtype=bool)
            for old_label in self.label_mapping.keys():
                unmapped_pixels &= (mask != old_label)
            mapped_mask[unmapped_pixels] = 0
            mask = mapped_mask
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask


def get_train_transforms():
    """Get training data augmentation transforms."""
    return A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=ROTATION_LIMIT, p=AUGMENTATION_PROB),
        A.RandomBrightnessContrast(
            brightness_limit=BRIGHTNESS_LIMIT,
            contrast_limit=CONTRAST_LIMIT,
            p=AUGMENTATION_PROB
        ),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_val_transforms():
    """Get validation data transforms."""
    return A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def create_dataloaders(dataset, label_mapping=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Create training and validation dataloaders.
    
    Args:
        dataset: The loaded dataset
        label_mapping: Label mapping dictionary for remapping labels
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Create datasets
    if 'validation' in dataset:
        train_dataset = FoodSegDataset(dataset['train'], transform=train_transform, label_mapping=label_mapping)
        val_dataset = FoodSegDataset(dataset['validation'], transform=val_transform, label_mapping=label_mapping)
    else:
        # Split train set if no validation set exists
        train_size = int((1 - VAL_SPLIT) * len(dataset['train']))
        val_size = len(dataset['train']) - train_size
        
        train_split, val_split = torch.utils.data.random_split(
            dataset['train'], [train_size, val_size]
        )
        
        train_dataset = FoodSegDataset(train_split, transform=train_transform, label_mapping=label_mapping)
        val_dataset = FoodSegDataset(val_split, transform=val_transform, label_mapping=label_mapping)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, val_loader


def get_class_names():
    """Get FoodSeg103 class names."""
    # This would typically be loaded from the dataset metadata
    # For now, we'll create a placeholder list
    class_names = [f"food_class_{i}" for i in range(103)]
    class_names.insert(0, "background")  # Add background class
    return class_names


if __name__ == "__main__":
    # Test dataset loading
    dataset, label_mapping = prepare_dataset()
    
    # Test dataloader creation
    train_loader, val_loader = create_dataloaders(dataset, label_mapping, batch_size=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test batch loading
    for images, masks in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Mask dtype: {masks.dtype}")
        print(f"Mask unique values: {torch.unique(masks)}")
        break

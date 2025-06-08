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
import hashlib
import pickle
import json
from config import *


def generate_cache_key(desired_labels, min_label_pixels, remap_labels):
    """
    生成基于筛选参数的缓存键。
    
    Args:
        desired_labels: 需要保留的标签ID列表
        min_label_pixels: 最小像素阈值
        remap_labels: 是否重新映射标签
    
    Returns:
        str: 缓存键字符串
    """
    # 将参数转换为字符串并计算哈希
    params_str = f"{sorted(desired_labels) if desired_labels else 'all'}_{min_label_pixels}_{remap_labels}"
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    return cache_key


def save_filtered_dataset_cache(filtered_dataset, label_mapping, cache_key):
    """
    保存筛选后的数据集到缓存。
    
    Args:
        filtered_dataset: 筛选后的数据集
        label_mapping: 标签映射字典
        cache_key: 缓存键
    """
    if not ENABLE_DATASET_CACHE:
        return
    
    cache_dir = FILTERED_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_data = {
        'filtered_dataset': filtered_dataset,
        'label_mapping': label_mapping,
        'cache_key': cache_key,
        'desired_labels': DESIRED_LABELS,
        'min_label_pixels': MIN_LABEL_PIXELS,
        'remap_labels': REMAP_LABELS
    }
    
    cache_file = os.path.join(cache_dir, f"filtered_dataset_{cache_key}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"筛选后的数据集已缓存到: {cache_file}")
    except Exception as e:
        print(f"保存数据集缓存失败: {str(e)}")


def load_filtered_dataset_cache(cache_key):
    """
    从缓存加载筛选后的数据集。
    
    Args:
        cache_key: 缓存键
    
    Returns:
        tuple: (filtered_dataset, label_mapping) 或 (None, None) 如果缓存不存在
    """
    if not ENABLE_DATASET_CACHE:
        return None, None
    
    cache_dir = FILTERED_CACHE_DIR
    cache_file = os.path.join(cache_dir, f"filtered_dataset_{cache_key}.pkl")
    
    if not os.path.exists(cache_file):
        return None, None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 验证缓存的参数是否匹配当前配置
        if (cache_data.get('desired_labels') == DESIRED_LABELS and
            cache_data.get('min_label_pixels') == MIN_LABEL_PIXELS and
            cache_data.get('remap_labels') == REMAP_LABELS):
            
            print(f"从缓存加载筛选后的数据集: {cache_file}")
            return cache_data['filtered_dataset'], cache_data['label_mapping']
        else:
            print("缓存参数不匹配当前配置，将重新筛选数据集")
            return None, None
            
    except Exception as e:
        print(f"加载数据集缓存失败: {str(e)}")
        return None, None


def filter_dataset_by_labels(dataset_split, desired_labels=None, min_label_pixels=100):
    """
    筛选数据集，只保留包含指定标签的样本（单线程版本）。
    
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
    
    # 逐个处理样本
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
            label_counts = {}
            
            for label in unique_labels:
                if label in desired_labels and label != 0:  # 跳过背景标签（0）
                    label_pixels = np.sum(mask == label)
                    if label_pixels >= min_label_pixels:
                        has_desired_label = True
                        # 记录这个标签在当前样本中出现
                        label_counts[label] = 1
            
            if has_desired_label:
                filtered_indices.append(idx)
                
                # 更新标签统计
                for label in label_counts:
                    if label not in label_stats:
                        label_stats[label] = 0
                    label_stats[label] += 1
                    
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
        # 背景标签（0）保持为0，指定保留的标签按顺序映射为1到标签数量
        # 其他未指定的标签都映射为背景（0）
        
        # 只为实际出现且在desired_labels中的标签创建映射
        present_desired_labels = [label for label in desired_labels if label in label_stats]
        present_desired_labels = sorted(present_desired_labels)
        
        # 创建映射：背景（0）->0，其他指定标签按顺序映射为1,2,3...
        label_mapping = {}
        
        # 背景标签总是映射为0（即使不在desired_labels中）
        label_mapping[0] = 0
        
        # 其他指定保留的标签按顺序映射为1,2,3...
        new_label_id = 1
        for old_label in present_desired_labels:
            if old_label != 0:  # 跳过背景标签
                label_mapping[old_label] = new_label_id
                new_label_id += 1
        
        print(f"\n标签映射: {label_mapping}")
        print(f"映射后的类别数量: {len(label_mapping)} (包括背景)")
        print(f"指定保留的标签: {present_desired_labels}")
        
        # 所有其他标签（未在desired_labels中的）将在数据集中被映射为背景（0）
    
    return filtered_indices, label_mapping


def prepare_dataset():
    """Download and prepare the FoodSeg103 dataset."""
    print("Loading FoodSeg103 dataset...")
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FILTERED_CACHE_DIR, exist_ok=True)
    
    # 生成缓存键
    cache_key = generate_cache_key(DESIRED_LABELS, MIN_LABEL_PIXELS, REMAP_LABELS)
    print(f"数据集筛选缓存键: {cache_key}")
    
    # 尝试从缓存加载筛选后的数据集
    filtered_dataset, global_label_mapping = load_filtered_dataset_cache(cache_key)
    
    if filtered_dataset is not None:
        print("成功从缓存加载筛选后的数据集!")
        
        # 显示数据集信息
        for split_name, split_data in filtered_dataset.items():
            print(f"{split_name} 样本数: {len(split_data)}")
        
        if global_label_mapping is not None:
            print(f"标签映射: {global_label_mapping}")
            print(f"类别数量: {len(global_label_mapping)} (包括背景)")
        
        return filtered_dataset, global_label_mapping
    
    # 缓存不存在，需要重新筛选
    print("缓存不存在或已过期，开始重新筛选数据集...")
    
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
        print(f"更新后的类别数量: {NUM_CLASSES} (包括背景)")
        
        # 显示最终的标签映射信息
        print(f"最终标签映射: {global_label_mapping}")
        background_count = sum(1 for v in global_label_mapping.values() if v == 0)
        food_count = len(global_label_mapping) - background_count
        print(f"背景类: {background_count}个, 食物类: {food_count}个")
    
    # 保存筛选后的数据集到缓存
    save_filtered_dataset_cache(filtered_dataset, global_label_mapping, cache_key)
    
    return filtered_dataset, global_label_mapping


def clear_dataset_cache(cache_key=None):
    """
    清理数据集缓存。
    
    Args:
        cache_key: 特定的缓存键，None表示清理所有缓存
    """
    if not ENABLE_DATASET_CACHE:
        print("数据集缓存功能未启用")
        return
    
    cache_dir = FILTERED_CACHE_DIR
    if not os.path.exists(cache_dir):
        print("缓存目录不存在")
        return
    
    if cache_key is None:
        # 清理所有缓存
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith("filtered_dataset_") and f.endswith(".pkl")]
        for cache_file in cache_files:
            cache_path = os.path.join(cache_dir, cache_file)
            try:
                os.remove(cache_path)
                print(f"已删除缓存文件: {cache_file}")
            except Exception as e:
                print(f"删除缓存文件失败 {cache_file}: {str(e)}")
        
        if not cache_files:
            print("没有找到缓存文件")
        else:
            print(f"共删除 {len(cache_files)} 个缓存文件")
    else:
        # 清理特定缓存
        cache_file = os.path.join(cache_dir, f"filtered_dataset_{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"已删除缓存文件: filtered_dataset_{cache_key}.pkl")
            except Exception as e:
                print(f"删除缓存文件失败: {str(e)}")
        else:
            print(f"缓存文件不存在: filtered_dataset_{cache_key}.pkl")


def list_dataset_cache():
    """
    列出所有可用的数据集缓存。
    """
    cache_dir = FILTERED_CACHE_DIR
    if not os.path.exists(cache_dir):
        print("缓存目录不存在")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith("filtered_dataset_") and f.endswith(".pkl")]
    
    if not cache_files:
        print("没有找到缓存文件")
        return
    
    print(f"找到 {len(cache_files)} 个缓存文件:")
    for cache_file in cache_files:
        cache_path = os.path.join(cache_dir, cache_file)
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            cache_key = cache_data.get('cache_key', 'unknown')
            desired_labels = cache_data.get('desired_labels', 'unknown')
            min_label_pixels = cache_data.get('min_label_pixels', 'unknown')
            remap_labels = cache_data.get('remap_labels', 'unknown')
            
            # 获取文件大小
            file_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
            
            print(f"  {cache_file}:")
            print(f"    缓存键: {cache_key}")
            print(f"    筛选标签: {desired_labels}")
            print(f"    最小像素: {min_label_pixels}")
            print(f"    重映射标签: {remap_labels}")
            print(f"    文件大小: {file_size:.2f} MB")
            
            if 'filtered_dataset' in cache_data:
                dataset_info = cache_data['filtered_dataset']
                for split_name, split_data in dataset_info.items():
                    print(f"    {split_name}: {len(split_data)} 样本")
            print()
            
        except Exception as e:
            print(f"  {cache_file}: 读取失败 - {str(e)}")


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
            mapped_mask = np.zeros_like(mask)  # 默认所有像素都是背景（0）
            
            # 应用标签映射
            for old_label, new_label in self.label_mapping.items():
                mapped_mask[mask == old_label] = new_label
            
            # 注意：未在label_mapping中的标签会保持为背景（0）
            # 这样就实现了"其他未指定的标签都映射为背景"的需求
            mask = mapped_mask
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to long tensor for CrossEntropyLoss
        if isinstance(mask, torch.Tensor):
            mask = mask.clone().detach().long()
        else:
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
    print("=== 数据集缓存测试 ===")
    
    # 显示当前配置
    cache_key = generate_cache_key(DESIRED_LABELS, MIN_LABEL_PIXELS, REMAP_LABELS)
    print(f"当前配置的缓存键: {cache_key}")
    print(f"缓存功能启用: {ENABLE_DATASET_CACHE}")
    
    # 列出现有缓存
    print("\n现有缓存:")
    list_dataset_cache()
    
    # 加载数据集（会自动使用缓存或创建新缓存）
    print("\n=== 加载数据集 ===")
    dataset, label_mapping = prepare_dataset()
    
    # Test dataloader creation
    print("\n=== 测试数据加载器 ===")
    train_loader, val_loader = create_dataloaders(dataset, label_mapping, batch_size=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test batch loading
    print("\n=== 测试批次加载 ===")
    for images, masks in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Mask dtype: {masks.dtype}")
        print(f"Mask unique values: {torch.unique(masks)}")
        break
    
    print("\n=== 缓存测试完成 ===")
    print("提示: 再次运行此脚本应该会从缓存加载数据集")

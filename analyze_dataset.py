#!/usr/bin/env python3

"""
数据集分析脚本：分析FoodSeg103数据集的标签分布
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from PIL import Image
import json

from dataset import prepare_dataset
from config import *


def analyze_dataset_labels(dataset_split, split_name="train", max_samples=None):
    """
    分析数据集中的标签分布
    
    Args:
        dataset_split: 数据集分割
        split_name: 分割名称
        max_samples: 最大分析样本数，None表示分析所有样本
    
    Returns:
        dict: 包含标签统计信息的字典
    """
    print(f"\n分析 {split_name} 数据集标签分布...")
    
    label_counts = Counter()
    label_pixel_counts = Counter()
    total_pixels = 0
    sample_count = 0
    
    # 限制分析的样本数
    num_samples = len(dataset_split) if max_samples is None else min(max_samples, len(dataset_split))
    
    for idx in tqdm(range(num_samples), desc=f"分析{split_name}数据"):
        try:
            sample = dataset_split[idx]
            
            # 获取mask
            mask = sample['label'] if 'label' in sample else sample['mask']
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            
            # 确保mask是单通道
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            # 统计标签
            unique_labels, counts = np.unique(mask, return_counts=True)
            
            for label, count in zip(unique_labels, counts):
                label_counts[int(label)] += 1  # 出现在多少个样本中
                label_pixel_counts[int(label)] += int(count)  # 总像素数
            
            total_pixels += mask.size
            sample_count += 1
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            continue
    
    # 计算统计信息
    stats = {
        'split_name': split_name,
        'total_samples': sample_count,
        'total_pixels': total_pixels,
        'unique_labels': len(label_counts),
        'label_counts': dict(label_counts),  # 每个标签出现在多少个样本中
        'label_pixel_counts': dict(label_pixel_counts),  # 每个标签的总像素数
        'label_frequencies': {},  # 每个标签的像素频率
        'sample_frequencies': {}  # 每个标签的样本频率
    }
    
    # 计算频率
    for label in label_counts.keys():
        stats['label_frequencies'][label] = label_pixel_counts[label] / total_pixels
        stats['sample_frequencies'][label] = label_counts[label] / sample_count
    
    return stats


def visualize_label_distribution(stats, save_dir="./analysis_results"):
    """可视化标签分布"""
    os.makedirs(save_dir, exist_ok=True)
    
    split_name = stats['split_name']
    
    # 1. 标签出现频率（样本维度）
    labels = list(stats['sample_frequencies'].keys())
    sample_freqs = list(stats['sample_frequencies'].values())
    
    # 按频率排序
    sorted_data = sorted(zip(labels, sample_freqs), key=lambda x: x[1], reverse=True)
    labels, sample_freqs = zip(*sorted_data)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    top_labels = labels[:20]  # 显示前20个最常见的标签
    top_freqs = sample_freqs[:20]
    
    plt.bar(range(len(top_labels)), top_freqs)
    plt.xlabel('标签ID')
    plt.ylabel('样本频率')
    plt.title(f'{split_name} - 前20个最常见标签（样本维度）')
    plt.xticks(range(len(top_labels)), top_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. 标签像素频率
    pixel_freqs = [stats['label_frequencies'][label] for label in labels]
    
    plt.subplot(1, 2, 2)
    top_pixel_freqs = pixel_freqs[:20]
    
    plt.bar(range(len(top_labels)), top_pixel_freqs)
    plt.xlabel('标签ID')
    plt.ylabel('像素频率')
    plt.title(f'{split_name} - 前20个最常见标签（像素维度）')
    plt.xticks(range(len(top_labels)), top_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{split_name}_label_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 标签分布直方图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(sample_freqs, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('样本频率')
    plt.ylabel('标签数量')
    plt.title(f'{split_name} - 样本频率分布')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(pixel_freqs, bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('像素频率')
    plt.ylabel('标签数量')
    plt.title(f'{split_name} - 像素频率分布')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    pixel_counts = list(stats['label_pixel_counts'].values())
    plt.hist(pixel_counts, bins=50, alpha=0.7, edgecolor='black', color='green')
    plt.xlabel('总像素数 (log scale)')
    plt.ylabel('标签数量')
    plt.title(f'{split_name} - 标签像素数分布')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    sample_counts = list(stats['label_counts'].values())
    plt.hist(sample_counts, bins=50, alpha=0.7, edgecolor='black', color='red')
    plt.xlabel('样本数量')
    plt.ylabel('标签数量')
    plt.title(f'{split_name} - 标签样本数分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{split_name}_distribution_histograms.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def recommend_label_filtering(stats, target_classes=20, min_sample_freq=0.01):
    """
    基于数据分析推荐标签筛选配置
    
    Args:
        stats: 数据集统计信息
        target_classes: 目标类别数
        min_sample_freq: 最小样本频率阈值
    
    Returns:
        dict: 推荐的配置
    """
    print(f"\n基于 {stats['split_name']} 数据集分析，推荐标签筛选配置...")
    
    # 按样本频率排序标签
    sorted_labels = sorted(stats['sample_frequencies'].items(), 
                          key=lambda x: x[1], reverse=True)
    
    # 筛选策略1：选择最常见的N个标签
    top_labels = [label for label, freq in sorted_labels[:target_classes]]
    
    # 筛选策略2：选择样本频率大于阈值的标签
    freq_filtered_labels = [label for label, freq in sorted_labels 
                           if freq >= min_sample_freq]
    
    # 筛选策略3：平衡像素频率和样本频率
    balanced_scores = {}
    for label in stats['sample_frequencies'].keys():
        sample_freq = stats['sample_frequencies'][label]
        pixel_freq = stats['label_frequencies'][label]
        # 综合得分：样本频率和像素频率的几何平均
        balanced_scores[label] = np.sqrt(sample_freq * pixel_freq)
    
    balanced_labels = sorted(balanced_scores.items(), key=lambda x: x[1], reverse=True)
    balanced_top_labels = [label for label, score in balanced_labels[:target_classes]]
    
    recommendations = {
        "strategy_1_top_frequent": {
            "DESIRED_LABELS": top_labels,
            "REMAP_LABELS": True,
            "MIN_LABEL_PIXELS": 100,
            "description": f"选择最常见的{len(top_labels)}个标签",
            "num_classes": len(top_labels)
        },
        
        "strategy_2_frequency_threshold": {
            "DESIRED_LABELS": freq_filtered_labels,
            "REMAP_LABELS": True, 
            "MIN_LABEL_PIXELS": 100,
            "description": f"选择样本频率>{min_sample_freq}的{len(freq_filtered_labels)}个标签",
            "num_classes": len(freq_filtered_labels)
        },
        
        "strategy_3_balanced": {
            "DESIRED_LABELS": balanced_top_labels,
            "REMAP_LABELS": True,
            "MIN_LABEL_PIXELS": 100,
            "description": f"平衡样本和像素频率的{len(balanced_top_labels)}个标签",
            "num_classes": len(balanced_top_labels)
        }
    }
    
    print("\n推荐配置:")
    for name, config in recommendations.items():
        print(f"\n{name}:")
        print(f"  描述: {config['description']}")
        print(f"  标签数: {config['num_classes']}")
        print(f"  前10个标签: {config['DESIRED_LABELS'][:10]}")
    
    return recommendations


def main():
    """主分析函数"""
    print("="*60)
    print("FoodSeg103 数据集标签分析")
    print("="*60)
    
    # 设置临时配置，先不进行标签筛选
    original_desired_labels = DESIRED_LABELS
    import config
    config.DESIRED_LABELS = None  # 临时设置为None以获取完整数据集
    
    try:
        # 加载数据集
        dataset, _ = prepare_dataset()
        
        analysis_results = {}
        recommendations = {}
        
        # 分析每个数据分割
        for split_name in dataset.keys():
            print(f"\n正在分析 {split_name} 数据集...")
            
            # 限制分析样本数以节省时间（可根据需要调整）
            max_samples = 1000 if len(dataset[split_name]) > 1000 else None
            
            stats = analyze_dataset_labels(dataset[split_name], split_name, max_samples)
            analysis_results[split_name] = stats
            
            # 可视化
            visualize_label_distribution(stats)
            
            # 生成推荐配置（仅对训练集）
            if split_name == 'train':
                recommendations = recommend_label_filtering(stats)
        
        # 保存分析结果
        results_dir = "./analysis_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存统计信息
        with open(os.path.join(results_dir, 'dataset_analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # 保存推荐配置
        with open(os.path.join(results_dir, 'recommended_configs.json'), 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"\n分析完成！结果保存在 {results_dir} 目录中")
        print("- dataset_analysis.json: 详细统计信息")
        print("- recommended_configs.json: 推荐的标签筛选配置")
        print("- *_label_distribution.png: 标签分布可视化")
        print("- *_distribution_histograms.png: 分布直方图")
        
    finally:
        # 恢复原始配置
        config.DESIRED_LABELS = original_desired_labels


if __name__ == "__main__":
    main()

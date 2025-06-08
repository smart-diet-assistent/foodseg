#!/usr/bin/env python3
"""
测试推理脚本
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 添加当前目录到Python路径
sys.path.append('/root/foodseg')

from model import create_model
from utils import create_colored_mask
from config import *
import albumentations as A
from albumentations.pytorch import ToTensorV2


def test_inference(image_path="image.jpg"):
    """测试推理功能"""
    print("🧪 测试推理功能")
    print("="*50)
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return False
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 加载模型
    model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"🔄 加载模型: {model_path}")
    model = create_model(NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ 模型加载成功")
    
    # 加载和预处理图像
    print(f"🖼️  加载图像: {image_path}")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ 无法读取图像")
        return False
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_shape = original_image.shape
    print(f"原始图像形状: {original_shape}")
    
    # 预处理
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    
    transformed = transform(image=original_image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 推理
    print("🔍 执行推理...")
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = outputs['out']
        pred_mask = torch.argmax(predictions, dim=1)
    
    print(f"预测输出形状: {pred_mask.shape}")
    
    # 后处理
    pred_mask_np = pred_mask.squeeze(0).cpu().numpy()
    print(f"预测掩码形状: {pred_mask_np.shape}")
    
    # 调整到原始尺寸
    if pred_mask_np.shape != original_shape[:2]:
        pred_mask_np = cv2.resize(
            pred_mask_np.astype(np.float32),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
    
    print(f"调整后掩码形状: {pred_mask_np.shape}")
    
    # 创建彩色掩码
    try:
        colored_mask = create_colored_mask(pred_mask_np, NUM_CLASSES)
        print(f"彩色掩码形状: {colored_mask.shape}")
    except Exception as e:
        print(f"⚠️  创建彩色掩码失败: {str(e)}")
        # 创建一个简单的彩色掩码
        colored_mask = np.zeros((*pred_mask_np.shape, 3))
        unique_classes = np.unique(pred_mask_np)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        for i, class_id in enumerate(unique_classes):
            mask = pred_mask_np == class_id
            colored_mask[mask] = colors[i][:3]
        print(f"简单彩色掩码形状: {colored_mask.shape}")
    
    # 测试图像叠加
    print("🎨 测试图像叠加...")
    try:
        # 确保尺寸匹配
        if colored_mask.shape[:2] != original_image.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (original_image.shape[1], original_image.shape[0]))
        
        # 确保数据类型匹配
        original_float = original_image.astype(np.float32)
        colored_float = (colored_mask * 255).astype(np.float32)
        
        print(f"叠加前 - 原始图像: {original_float.shape}, 彩色掩码: {colored_float.shape}")
        
        overlay = cv2.addWeighted(original_float, 0.7, colored_float, 0.3, 0)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        print(f"叠加后形状: {overlay.shape}")
        print("✅ 图像叠加成功")
        
    except Exception as e:
        print(f"❌ 图像叠加失败: {str(e)}")
        overlay = original_image
    
    # 分析结果
    unique_classes = np.unique(pred_mask_np)
    print(f"检测到的类别: {unique_classes}")
    
    total_pixels = pred_mask_np.size
    for class_id in unique_classes:
        if class_id == 0:
            continue
        pixels = np.sum(pred_mask_np == class_id)
        percentage = (pixels / total_pixels) * 100
        if percentage > 1.0:
            print(f"类别 {class_id}: {percentage:.2f}% ({pixels:,} 像素)")
    
    # 创建简单可视化
    print("📊 创建可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_mask_np, cmap='tab20')
    axes[0, 1].set_title('分割掩码')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(colored_mask)
    axes[1, 0].set_title('彩色掩码')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('叠加结果')
    axes[1, 1].axis('off')
    
    plt.suptitle('推理测试结果')
    plt.tight_layout()
    
    # 保存结果
    output_path = 'test_inference_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 结果已保存到: {output_path}")
    
    plt.show()
    
    print("✅ 推理测试完成!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "image.jpg"
    
    success = test_inference(image_path)
    if success:
        print("🎉 测试成功!")
    else:
        print("❌ 测试失败!")

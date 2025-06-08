#!/usr/bin/env python3
"""
食物分割推理脚本
对输入图片进行食物分割识别，输出识别结果、食物种类和面积信息
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import create_model
from utils import create_colored_mask
from config import *


# Food category name mapping (English version)
FOOD_NAMES = {
    0: "Background",
    1: "Chocolate",
    2: "Cheese/Butter", 
    3: "Milk",
    4: "Egg",
    5: "Apple",
    6: "Banana",
    7: "Strawberry",
    8: "Mango",
    9: "Orange",
    10: "Beef Steak",
    11: "Pork",
    12: "Chicken/Duck",
    13: "Sauce",
    14: "Bread",
    15: "Corn",
    16: "Pasta",
    17: "Noodles",
    18: "Rice",
    19: "Tofu",
    20: "Eggplant",
    21: "Potato",
    22: "Garlic",
    23: "Tomato",
    24: "Scallion",
    25: "Ginger",
    26: "Lettuce",
    27: "Cucumber",
    28: "Carrot",
    29: "Cabbage",
    30: "Green Bean",
    31: "King Oyster Mushroom",
    32: "Shiitake Mushroom"
}


def load_label_mapping():
    """加载标签映射文件"""
    label_mapping_file = os.path.join(DATA_DIR, 'label_mapping.json')
    if os.path.exists(label_mapping_file):
        with open(label_mapping_file, 'r') as f:
            return json.load(f)
    return None


def get_reverse_label_mapping(label_mapping):
    """获取反向标签映射（从映射后的ID到原始ID）"""
    if label_mapping is None:
        return None
    return {v: int(k) for k, v in label_mapping.items()}


def load_trained_model(model_path):
    """
    加载训练好的模型
    
    Args:
        model_path (str): 模型文件路径
    
    Returns:
        tuple: (model, device) 或 (None, None) 如果加载失败
    """
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 创建模型
    print("创建模型...")
    model = create_model(NUM_CLASSES, pretrained=False)
    
    # 加载检查点
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None
    
    try:
        print(f"加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 打印模型信息
        if 'epoch' in checkpoint:
            print(f"模型训练轮数: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"模型性能 - mIoU: {metrics.get('mean_iou', 'N/A'):.4f}")
            
        print("✅ 模型加载成功!")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None, None
    
    model = model.to(device)
    model.eval()
    
    return model, device


def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    预处理输入图像
    
    Args:
        image_path (str): 图像文件路径
        target_size (tuple): 目标尺寸 (height, width)
    
    Returns:
        tuple: (preprocessed_tensor, original_image, original_size)
    """
    # 读取图像
    if isinstance(image_path, str):
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_image = image_path
    
    original_size = original_image.shape[:2]  # (H, W)
    
    # 数据增强/预处理管道
    transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    
    # 应用变换
    transformed = transform(image=original_image)
    image_tensor = transformed['image'].unsqueeze(0)  # 添加batch维度
    
    return image_tensor, original_image, original_size


def postprocess_prediction(prediction, original_size):
    """
    后处理预测结果到原始图像尺寸
    
    Args:
        prediction (torch.Tensor): 模型预测结果
        original_size (tuple): 原始图像尺寸 (H, W)
    
    Returns:
        numpy.ndarray: 调整尺寸后的预测掩码
    """
    # 移除batch维度并转换为numpy
    if prediction.dim() == 4:
        prediction = prediction.squeeze(0)
    if prediction.dim() == 3:
        prediction = prediction.squeeze(0)
    
    prediction = prediction.cpu().numpy()
    
    print(f"调整前预测掩码形状: {prediction.shape}")
    print(f"目标尺寸: {original_size}")
    
    # 调整到原始尺寸
    if prediction.shape != original_size:
        # 确保prediction是2D的
        if len(prediction.shape) > 2:
            prediction = prediction.squeeze()
        
        prediction = cv2.resize(
            prediction.astype(np.float32), 
            (original_size[1], original_size[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    print(f"调整后预测掩码形状: {prediction.shape}")
    return prediction.astype(np.uint8)


def calculate_food_areas(pred_mask, pixel_size_mm2=None):
    """
    计算各个食物类别的面积
    
    Args:
        pred_mask (numpy.ndarray): 预测掩码
        pixel_size_mm2 (float): 每个像素代表的面积(平方毫米)，如果为None则返回像素数
    
    Returns:
        dict: 各类别的面积信息
    """
    unique_classes = np.unique(pred_mask)
    class_areas = {}
    
    total_pixels = pred_mask.size
    
    for class_id in unique_classes:
        if class_id == 0:  # 跳过背景
            continue
            
        class_pixels = np.sum(pred_mask == class_id)
        class_percentage = (class_pixels / total_pixels) * 100
        
        class_name = FOOD_NAMES.get(class_id, f"Unknown_Class_{class_id}")
        
        area_info = {
            'pixels': class_pixels,
            'percentage': class_percentage,
            'name': class_name
        }
        
        if pixel_size_mm2 is not None:
            area_info['area_mm2'] = class_pixels * pixel_size_mm2
            area_info['area_cm2'] = area_info['area_mm2'] / 100
        
        class_areas[class_id] = area_info
    
    return class_areas


def create_visualization(original_image, pred_mask, class_areas, save_path=None):
    """
    创建可视化结果
    
    Args:
        original_image (numpy.ndarray): 原始图像
        pred_mask (numpy.ndarray): 预测掩码
        class_areas (dict): 类别面积信息
        save_path (str): 保存路径，可选
    
    Returns:
        matplotlib.figure.Figure: 可视化图像
    """
    print(f"🎨 创建可视化 - 原始图像形状: {original_image.shape}")
    print(f"🎨 创建可视化 - 预测掩码形状: {pred_mask.shape}")
    
    # 确保预测掩码是2D的
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask.squeeze()
        print(f"🎨 压缩后掩码形状: {pred_mask.shape}")
    
    # 创建彩色掩码
    try:
        colored_mask = create_colored_mask(pred_mask, NUM_CLASSES)
        print(f"🎨 彩色掩码形状: {colored_mask.shape}")
    except Exception as e:
        print(f"❌ 创建彩色掩码失败: {str(e)}")
        # 创建一个简单的彩色掩码
        colored_mask = np.zeros((*pred_mask.shape, 3))
        for i in range(min(NUM_CLASSES, 20)):
            mask_i = (pred_mask == i)
            colored_mask[mask_i] = plt.cm.tab20(i % 20)[:3]
    
    # 确保所有图像具有相同的尺寸
    original_height, original_width = original_image.shape[:2]
    
    # 确保彩色掩码与原始图像尺寸一致
    if colored_mask.shape[:2] != (original_height, original_width):
        colored_mask = cv2.resize(colored_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        print(f"🎨 调整后彩色掩码形状: {colored_mask.shape}")
    
    # 确保预测掩码与原始图像尺寸一致
    if pred_mask.shape != (original_height, original_width):
        pred_mask = cv2.resize(pred_mask.astype(np.float32), (original_width, original_height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        print(f"🎨 调整后预测掩码形状: {pred_mask.shape}")
    
    # 创建叠加图像 - 使用更安全的方法
    try:
        # 确保两个图像都是正确的类型和尺寸
        original_for_overlay = original_image.astype(np.float32)
        colored_for_overlay = (colored_mask * 255).astype(np.float32)
        
        print(f"🎨 叠加前 - 原始图像: {original_for_overlay.shape}, 彩色掩码: {colored_for_overlay.shape}")
        
        # 确保两个图像具有相同的形状
        if original_for_overlay.shape != colored_for_overlay.shape:
            print(f"⚠️ 形状不匹配，尝试修复...")
            if len(original_for_overlay.shape) == 3 and len(colored_for_overlay.shape) == 3:
                if original_for_overlay.shape[2] != colored_for_overlay.shape[2]:
                    # 通道数不匹配
                    if colored_for_overlay.shape[2] == 1:
                        colored_for_overlay = np.repeat(colored_for_overlay, 3, axis=2)
                    elif original_for_overlay.shape[2] == 1:
                        original_for_overlay = np.repeat(original_for_overlay, 3, axis=2)
        
        # 再次检查形状
        if original_for_overlay.shape == colored_for_overlay.shape:
            overlay = cv2.addWeighted(original_for_overlay, 0.6, colored_for_overlay, 0.4, 0)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            print("✅ 成功创建叠加图像")
        else:
            print(f"❌ 形状仍然不匹配: {original_for_overlay.shape} vs {colored_for_overlay.shape}")
            overlay = original_image.copy()
    except Exception as e:
        print(f"❌ 创建叠加图像失败: {str(e)}")
        overlay = original_image.copy()
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Prediction mask
    plt.subplot(2, 3, 2)
    plt.imshow(pred_mask, cmap='tab20')
    plt.title('Segmentation Mask', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Colored mask
    plt.subplot(2, 3, 3)
    plt.imshow(colored_mask)
    plt.title('Colored Mask', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Overlay image
    plt.subplot(2, 3, 4)
    plt.imshow(overlay)
    plt.title('Overlay Result', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Area statistics chart
    plt.subplot(2, 3, 5)
    if class_areas:
        class_names = [info['name'] for info in class_areas.values()]
        percentages = [info['percentage'] for info in class_areas.values()]
        
        # Only show categories with area > 1%
        filtered_data = [(name, pct) for name, pct in zip(class_names, percentages) if pct > 1.0]
        
        if filtered_data:
            names, pcts = zip(*filtered_data)
            colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
            
            plt.pie(pcts, labels=names, autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Food Category Distribution', fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No significant food regions detected', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Food Category Distribution', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No food detected', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Food Category Distribution', fontsize=14, fontweight='bold')
    
    # Detailed statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    if class_areas:
        # Sort by area percentage
        sorted_areas = sorted(class_areas.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        info_text = "Detected Food Categories:\n" + "="*30 + "\n"
        for class_id, info in sorted_areas:
            if info['percentage'] > 0.5:  # Only show categories with area > 0.5%
                info_text += f"{info['name']}:\n"
                info_text += f"  Pixels: {info['pixels']:,}\n"
                info_text += f"  Percentage: {info['percentage']:.2f}%\n"
                if 'area_cm2' in info:
                    info_text += f"  Area: {info['area_cm2']:.2f} cm²\n"
                info_text += "\n"
        
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    else:
        plt.text(0.5, 0.5, 'No food categories detected', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold')
    
    plt.suptitle('Food Segmentation Recognition Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 结果已保存到: {save_path}")
    
    return fig


def inference_single_image(model, device, image_path, output_dir=None, show_result=True):
    """
    对单张图像进行推理
    
    Args:
        model: 训练好的模型
        device: 设备
        image_path (str): 图像路径
        output_dir (str): 输出目录，可选
        show_result (bool): 是否显示结果
    
    Returns:
        dict: 推理结果
    """
    print(f"正在处理图像: {image_path}")
    
    try:
        # 预处理
        image_tensor, original_image, original_size = preprocess_image(image_path)
        print(f"原始图像尺寸: {original_image.shape}")
        print(f"原始尺寸记录: {original_size}")
        image_tensor = image_tensor.to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = outputs['out']
            pred_mask = torch.argmax(predictions, dim=1)
            print(f"预测输出尺寸: {pred_mask.shape}")
        
        # 后处理
        pred_mask_np = postprocess_prediction(pred_mask, original_size)
        print(f"后处理后掩码尺寸: {pred_mask_np.shape}")
        
        # 计算面积
        class_areas = calculate_food_areas(pred_mask_np)
        
        # 创建输出文件名
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            result_path = os.path.join(output_dir, f"{base_name}_result.png")
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        else:
            result_path = None
            mask_path = None
        
        # 创建可视化
        fig = create_visualization(original_image, pred_mask_np, class_areas, result_path)
        
        # 保存掩码
        if mask_path:
            cv2.imwrite(mask_path, pred_mask_np)
            print(f"✅ 掩码已保存到: {mask_path}")
        
        # 显示结果
        if show_result:
            plt.show()
        else:
            plt.close(fig)
        
        # 打印统计信息
        print("\n" + "="*50)
        print("识别结果统计")
        print("="*50)
        
        if class_areas:
            sorted_areas = sorted(class_areas.items(), key=lambda x: x[1]['percentage'], reverse=True)
            for class_id, info in sorted_areas:
                if info['percentage'] > 0.5:
                    print(f"{info['name']}: {info['percentage']:.2f}% ({info['pixels']:,} 像素)")
        else:
            print("未检测到食物类别")
        
        print("="*50)
        
        return {
            'original_image': original_image,
            'pred_mask': pred_mask_np,
            'class_areas': class_areas,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ 处理图像时出错: {str(e)}")
        return {'success': False, 'error': str(e)}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='食物分割推理脚本')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--model', '-m', type=str, 
                       default=os.path.join(MODEL_SAVE_DIR, 'best_model.pth'),
                       help='模型文件路径')
    parser.add_argument('--output', '-o', type=str, default='./inference_output',
                       help='输出目录')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示结果图像')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"❌ 输入图像不存在: {args.image}")
        return
    
    # 加载模型
    print("="*60)
    print("食物分割推理系统")
    print("="*60)
    
    model, device = load_trained_model(args.model)
    if model is None:
        return
    
    # 执行推理
    result = inference_single_image(
        model, device, args.image, 
        output_dir=args.output,
        show_result=not args.no_show
    )
    
    if result['success']:
        print("✅ 推理完成!")
    else:
        print(f"❌ 推理失败: {result['error']}")


if __name__ == "__main__":
    # 如果直接运行，检查是否有命令行参数
    if len(sys.argv) == 1:
        # 没有参数时的演示模式
        print("="*60)
        print("食物分割推理演示")
        print("="*60)
        print("用法示例:")
        print("python inference_demo.py --image /path/to/image.jpg")
        print("python inference_demo.py --image image.jpg --output ./results")
        print("python inference_demo.py --image image.jpg --model models/best_model.pth --no-show")
        
        # 检查是否有示例图像
        if os.path.exists('image.jpg'):
            print(f"\n检测到示例图像: image.jpg")
            response = input("是否使用此图像进行演示? (y/n): ")
            if response.lower() == 'y':
                sys.argv = ['inference_demo.py', '--image', 'image.jpg']
                main()
        
    else:
        main()

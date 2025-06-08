"""
LRASPP MobileNetV3-Large 输入规格说明和测试

这个脚本展示了 lraspp_mobilenet_v3_large 模型的输入要求和使用方法
"""

import torch
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation.lraspp import LRASPP_MobileNet_V3_Large_Weights
import numpy as np

def test_lraspp_inputs():
    """测试 LRASPP 模型的各种输入格式"""
    
    print("=" * 60)
    print("LRASPP MobileNetV3-Large 输入规格测试")
    print("=" * 60)
    
    # 1. 创建模型
    print("1. 创建预训练模型...")
    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.eval()
    
    print(f"   ✅ 模型创建成功")
    print(f"   📊 原始类别数: {model.classifier.high_classifier.out_channels}")
    
    # 2. 输入规格说明
    print("\n2. 输入张量规格:")
    print("   📋 格式: (batch_size, channels, height, width)")
    print("   📋 通道数: 3 (RGB)")
    print("   📋 数据类型: torch.float32")
    print("   📋 数值范围: [0, 1] (经过归一化)")
    print("   📋 推荐尺寸: 可变，但建议 >= 224x224")
    
    # 3. 标准化参数
    print("\n3. 预处理要求:")
    print("   🎨 ImageNet 标准化:")
    print("   📊 Mean: [0.485, 0.456, 0.406]")
    print("   📊 Std:  [0.229, 0.224, 0.225]")
    
    # 4. 测试不同尺寸的输入
    test_sizes = [
        (224, 224),   # 最小推荐尺寸
        (256, 256),   # 常用尺寸
        (384, 384),   # 中等尺寸
        (512, 512),   # 项目使用尺寸
        (640, 480),   # 不同宽高比
        (768, 768),   # 大尺寸
    ]
    
    print("\n4. 测试不同输入尺寸:")
    for i, (h, w) in enumerate(test_sizes):
        try:
            # 创建随机输入 (标准化后的)
            x = torch.randn(1, 3, h, w)  # 模拟标准化后的图像
            
            with torch.no_grad():
                output = model(x)
                out_shape = output['out'].shape
                
            print(f"   ✅ {h}x{w}: 输入 {x.shape} → 输出 {out_shape}")
            
        except Exception as e:
            print(f"   ❌ {h}x{w}: 错误 - {str(e)}")
    
    # 5. 批处理测试
    print("\n5. 测试不同批量大小:")
    batch_sizes = [1, 2, 4, 8, 16]
    input_size = (512, 512)
    
    for batch_size in batch_sizes:
        try:
            x = torch.randn(batch_size, 3, *input_size)
            
            with torch.no_grad():
                output = model(x)
                out_shape = output['out'].shape
                
            print(f"   ✅ Batch {batch_size}: 输入 {x.shape} → 输出 {out_shape}")
            
        except Exception as e:
            print(f"   ❌ Batch {batch_size}: 错误 - {str(e)}")
    
    # 6. 输出格式说明
    print("\n6. 输出格式:")
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
    
    print(f"   📤 输出类型: {type(output)}")
    print(f"   📤 输出键: {list(output.keys())}")
    print(f"   📤 'out' 形状: {output['out'].shape}")
    print(f"   📤 'out' 数据类型: {output['out'].dtype}")
    print(f"   📤 分割图形状: (batch, classes, height, width)")
    
    # 7. 预处理示例
    print("\n7. 完整预处理示例:")
    
    # 模拟原始图像 (0-255)
    raw_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    print(f"   📷 原始图像: {raw_image.shape}, 范围 [{raw_image.min()}, {raw_image.max()}]")
    
    # 转换为浮点数并归一化到 [0, 1]
    image_float = raw_image.astype(np.float32) / 255.0
    print(f"   🔄 归一化: {image_float.shape}, 范围 [{image_float.min():.3f}, {image_float.max():.3f}]")
    
    # ImageNet 标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_float - mean) / std
    print(f"   📊 标准化: {image_norm.shape}, 范围 [{image_norm.min():.3f}, {image_norm.max():.3f}]")
    
    # 转换为 PyTorch 张量 (HWC → CHW)
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
    print(f"   🔄 张量格式: {image_tensor.shape}, 数据类型: {image_tensor.dtype}")
    
    # 添加批次维度
    image_batch = image_tensor.unsqueeze(0)
    print(f"   📦 批次格式: {image_batch.shape}, 数据类型: {image_batch.dtype}")
    
    # 测试模型推理
    with torch.no_grad():
        result = model(image_batch)
        predictions = torch.argmax(result['out'], dim=1)
        
    print(f"   🎯 推理结果: {result['out'].shape}")
    print(f"   🏷️  预测类别: {predictions.shape}, 唯一值: {torch.unique(predictions).tolist()}")
    
    # 8. 常见错误和解决方案
    print("\n8. 常见错误和解决方案:")
    print("   ❌ 错误: 输入不是 float 类型")
    print("      ✅ 解决: image.float() 或 image.to(torch.float32)")
    print("   ❌ 错误: 数据类型不匹配 (Double vs Float)")
    print("      ✅ 解决: 使用 torch.from_numpy(array).float()")
    print("   ❌ 错误: 输入没有批次维度")
    print("      ✅ 解决: image.unsqueeze(0)")
    print("   ❌ 错误: 通道顺序错误 (HWC)")
    print("      ✅ 解决: image.permute(2, 0, 1) 或 numpy.transpose(2, 0, 1)")
    print("   ❌ 错误: 没有标准化")
    print("      ✅ 解决: 使用 ImageNet 的 mean 和 std")
    print("   ❌ 错误: 值域错误 (0-255)")
    print("      ✅ 解决: 先除以 255.0 再标准化")

def create_preprocessing_function():
    """创建标准的预处理函数"""
    
    def preprocess_image(image):
        """
        标准预处理函数
        
        Args:
            image: numpy array, shape (H, W, 3), dtype uint8, range [0, 255]
                  或 PIL Image
        
        Returns:
            torch.Tensor: shape (1, 3, H, W), dtype float32, 标准化后
        """
        # 如果是 PIL Image，转换为 numpy
        if hasattr(image, 'mode'):  # PIL Image
            image = np.array(image)
        
        # 确保是 RGB 格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            pass  # 已经是 RGB
        elif len(image.shape) == 2:
            # 灰度图转 RGB
            image = np.stack([image] * 3, axis=2)
        else:
            raise ValueError(f"不支持的图像格式: {image.shape}")
        
        # 转换为浮点数并归一化到 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype in [np.float32, np.float64]:
            if image.max() > 1.0:
                image = image / 255.0
        
        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # 转换为 PyTorch 张量 (HWC → CHW)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # 添加批次维度
        image_batch = image_tensor.unsqueeze(0)
        
        return image_batch
    
    return preprocess_image

if __name__ == "__main__":
    # 运行测试
    test_lraspp_inputs()
    
    print("\n" + "="*60)
    print("预处理函数示例")
    print("="*60)
    
    # 创建预处理函数
    preprocess = create_preprocessing_function()
    
    # 测试预处理函数
    raw_image = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    processed = preprocess(raw_image)
    print(f"原始图像: {raw_image.shape}, {raw_image.dtype}")
    print(f"处理后: {processed.shape}, {processed.dtype}")
    print(f"数值范围: [{processed.min():.3f}, {processed.max():.3f}]")
    
    print("\n🎉 测试完成！")

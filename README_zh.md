# FoodSeg103 语义分割 - LRASPP模型

> [English](README.md) | 简体中文

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目基于FoodSeg103数据集实现语义分割，使用LRASPP（轻量级减少型空洞空间金字塔池化）模型，具备先进的数据过滤功能。

## 数据集
- **FoodSeg103**: 包含103个食物类别的大规模食物图像分割数据集
- **来源**: Hugging Face上的EduardoPacheco/FoodSeg103
- **特性**: 支持标签过滤和类别子集选择

## 模型
- **LRASPP**: 轻量级减少型空洞空间金字塔池化
- **主干网络**: MobileNetV3-Large
- **用途**: 针对移动应用的轻量级语义分割

## 核心特性

### 🎯 标签过滤系统
- **选择性训练**: 选择特定的食物类别进行训练
- **标签重映射**: 自动将标签重映射为连续ID (0, 1, 2, ...)
- **像素阈值**: 过滤掉标签像素不足的样本
- **自动分析**: 内置数据集分析与建议功能

### 📊 数据集分析
- 全面的标签分布分析
- 自动过滤建议
- 可视化统计和直方图
- 分析结果导出为JSON格式

## 安装配置

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

2. 分析数据集（可选但推荐）：
```bash
python analyze_dataset.py
```

3. 在 `config.py` 中配置标签过滤：
```python
# 示例：仅训练前20个食物类别
DESIRED_LABELS = list(range(0, 21))  # 0=背景, 1-20=食物类别
REMAP_LABELS = True
MIN_LABEL_PIXELS = 100
```

4. 下载并准备数据集：
```bash
python prepare_dataset.py
```

5. 训练模型：
```bash
python train.py
```

6. 评估模型：
```bash
python evaluate.py
```

7. 运行推理：
```bash
python inference.py --image_path path/to/image.jpg
```

## 标签过滤配置

### 配置选项

```python
# 在 config.py 中

# 选项1：训练特定标签
DESIRED_LABELS = [0, 1, 2, 5, 10, 15, 20]  # 背景 + 6个食物类别

# 选项2：训练前N个类别
DESIRED_LABELS = list(range(0, 21))  # 背景 + 20个食物类别

# 选项3：使用所有类别（不过滤）
DESIRED_LABELS = None

# 标签重映射（推荐：True）
REMAP_LABELS = True

# 样本中每个标签的最小像素数
MIN_LABEL_PIXELS = 100
```

### 推荐工作流

1. **快速开始（小子集）**:
   ```python
   DESIRED_LABELS = [0, 1, 2, 3, 4, 5]  # 6个类别
   REMAP_LABELS = True
   MIN_LABEL_PIXELS = 100
   ```

2. **平衡训练（中等子集）**:
   ```python
   DESIRED_LABELS = list(range(0, 21))  # 21个类别
   REMAP_LABELS = True
   MIN_LABEL_PIXELS = 100
   ```

3. **完整数据集**:
   ```python
   DESIRED_LABELS = None  # 全部104个类别
   REMAP_LABELS = False
   MIN_LABEL_PIXELS = 50
   ```

## 项目结构
```
foodseg/
├── requirements.txt              # 依赖包
├── config.py                    # 配置文件（包含过滤选项）
├── dataset.py                   # 数据集加载（支持过滤）
├── model.py                     # LRASPP模型实现
├── train.py                     # 训练脚本
├── evaluate.py                  # 评估脚本
├── inference.py                 # 推理脚本
├── utils.py                     # 工具函数
├── prepare_dataset.py           # 数据集准备
├── analyze_dataset.py           # 数据集分析工具
├── label_filtering_examples.py  # 配置示例
├── README.md                    # 英文说明文档
├── README_zh.md                 # 中文说明文档
├── LICENSE                      # MIT许可证
├── wandb_config.py              # Weights & Biases配置
├── cache_manager.py             # 缓存管理工具
├── CACHE_README.md              # 缓存系统文档
├── convert_to_tflite.py         # TensorFlow Lite转换
├── test_inference.py            # 推理测试
├── test_lraspp_inputs.py        # 模型输入测试
└── inference_demo.py            # 交互式推理演示

# 生成的目录：
├── data/                        # 数据集缓存
│   └── label_mapping.json      # 标签映射文件
├── cache/                       # Hugging Face数据集缓存
│   └── filtered_datasets/      # 过滤后的数据集缓存
├── models/                      # 保存的模型
│   ├── best_model.pth          # PyTorch模型
│   ├── food_segmentation_model.onnx  # ONNX格式
│   ├── food_segmentation_model.tflite # TensorFlow Lite
│   └── food_segmentation_model_saved_model/ # TensorFlow SavedModel
├── results/                     # 训练结果和可视化
├── logs/                        # 训练日志
├── analysis_results/            # 数据集分析结果
├── inference_output/            # 推理输出图像
├── esp32_deployment/            # ESP32部署文件
└── FoodSegmentationLibrary/     # ESP32的Arduino库
```

## 模型导出与部署

### ONNX导出
```bash
# 将训练好的模型转换为ONNX格式
python convert_to_onnx.py --model_path models/best_model.pth
```

### TensorFlow Lite导出
```bash
# 转换为TensorFlow Lite以用于移动端部署
python convert_to_tflite.py --model_path models/best_model.pth
```

### ESP32部署
项目包含ESP32部署支持：
- `esp32_deployment/model_data.cc`中的转换模型权重
- `FoodSegmentationLibrary/`中的Arduino库
- 可直接用于微控制器部署

### 交互式演示
```bash
# 运行交互式推理演示
python inference_demo.py
```

## 训练与评估

### 训练结果
训练完成后，您可以在`results/`目录中找到各种输出：
- `best_predictions_epoch_*.png`: 每个epoch的最佳预测可视化
- `comprehensive_evaluation_results.txt`: 详细评估指标
- `comprehensive_metrics.png`: 性能可视化
- `confusion_matrix.png`: 类别混淆矩阵
- `evaluation_metrics.npz`: 数值评估数据

### 训练监控
项目支持Weights & Biases集成：
```bash
# 配置W&B（可选）
python wandb_config.py

# 使用W&B日志记录进行训练
python train.py --use_wandb
```

### 缓存管理
高效的数据集缓存系统：
```bash
# 检查缓存状态
python cache_manager.py --status

# 清除特定缓存
python cache_manager.py --clear filtered

# 详细缓存管理请参阅CACHE_README.md
```

## 测试

### 模型测试
```bash
# 测试LRASPP模型输入
python test_lraspp_inputs.py

# 测试推理流水线
python test_inference.py --image_path image.jpg
```

## 高级用法

### 数据集分析
```bash
# 分析标签分布并获取建议
python analyze_dataset.py
```
这将生成：
- `analysis_results/dataset_analysis.json`: 详细统计信息
- `analysis_results/recommended_configs.json`: 过滤建议
- `analysis_results/*_distribution.png`: 可视化图表

### 自定义标签选择
```python
# 示例：专注于常见食物
DESIRED_LABELS = [
    0,   # 背景
    1,   # 苹果
    5,   # 香蕉
    10,  # 面包
    15,  # 披萨
    20,  # 汉堡
    # ... 根据需要添加更多
]
```

### 批量推理
```bash
# 处理目录中的所有图像
python inference.py --image_path ./test_images/ --batch --output_dir ./results/
```

## 标签过滤的优势

1. **🚀 训练加速**: 减少的数据集规模意味着更快的迭代
2. **💾 内存效率**: 较小的类别集合需要更少的内存
3. **🎯 专注学习**: 在选定类别上获得更好的性能
4. **⚖️ 类别平衡**: 避免类别不平衡问题
5. **🔧 易于调试**: 较少的类别使分析和调试更简单

## 性能优化建议

1. **从小开始**: 初始实验使用5-10个类别
2. **使用分析**: 运行 `analyze_dataset.py` 来了解你的数据
3. **启用重映射**: 对于过滤的数据集始终设置 `REMAP_LABELS=True`
4. **调整阈值**: 根据目标对象大小调整 `MIN_LABEL_PIXELS`
5. **监控指标**: 检查各类别的IoU以识别问题类别

## 故障排除

### 常见问题
- **过滤后数据集为空**: 检查 `DESIRED_LABELS` 是否在数据集中存在
- **内存错误**: 减少批量大小或类别数量
- **性能差**: 确保每个类别有足够的样本（推荐>100个）

### 调试命令
```bash
# 测试数据集加载
python dataset.py

# 检查模型架构
python model.py

# 验证配置
python -c "from config import *; print(f'Classes: {NUM_CLASSES}, Labels: {DESIRED_LABELS}')"
```

## 常见问题

### 问：如何为我的使用场景选择合适的类别数量？
答：从5-10个类别开始实验，然后逐步增加。使用`analyze_dataset.py`了解类别分布和平衡性。

### 问：REMAP_LABELS=True和False有什么区别？
答：设为True时，标签会重新映射为连续ID（0,1,2,3...）。设为False时，保留原始标签ID。对于过滤的数据集，始终使用True。

### 问：我的训练很慢，如何加速？
答：
- 使用标签过滤减少类别数量
- 如果内存有限，使用较小的批量大小
- 启用混合精度训练
- 使用过滤数据集减少数据加载时间

### 问：我可以将此项目用于其他分割数据集吗？
答：可以，修改`dataset.py`中的数据集加载部分以支持您的数据格式。LRASPP模型可以用于任何语义分割任务。

### 问：如何将模型部署到移动设备？
答：使用TensorFlow Lite转换脚本`convert_to_tflite.py`创建移动端优化模型，然后集成到您的移动应用中。

## 环境要求

### 硬件要求
- Python 3.8或更高版本
- 支持CUDA的GPU（推荐用于训练）
- 完整数据集需要8GB以上内存
- 过滤数据集需要4GB以上内存

### 依赖包
```bash
# 核心依赖
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
datasets>=2.0.0
Pillow>=8.0.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0

# 可选依赖
onnx>=1.12.0           # 用于ONNX导出
tensorflow>=2.8.0      # 用于TensorFlow Lite导出
wandb>=0.12.0          # 用于实验跟踪
```

## 许可证

此项目基于MIT许可证开源。详情请参阅 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交问题和拉取请求！请确保遵循项目的代码规范。

## 致谢

- FoodSeg103数据集的创建者
- PyTorch团队提供的LRASPP实现
- Hugging Face提供的数据集托管服务

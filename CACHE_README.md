# 数据集缓存功能说明

## 功能概述

为了避免每次运行都重新筛选数据集，我添加了数据集缓存功能。该功能会：

1. 基于筛选参数（筛选标签、最小像素阈值、标签重映射设置）生成唯一的缓存键
2. 在首次筛选后将结果保存到缓存文件
3. 后续运行时自动检测并加载匹配的缓存
4. 当筛选参数改变时自动重新筛选并创建新缓存

## 配置选项

在 `config.py` 中添加了以下配置：

```python
# Dataset filtering cache configuration
FILTERED_CACHE_DIR = "./cache/filtered_datasets"  # 缓存目录
ENABLE_DATASET_CACHE = True  # 是否启用数据集缓存
```

## 使用方法

### 1. 自动缓存使用

正常运行你的代码，缓存功能会自动工作：

```bash
python dataset.py  # 第一次运行会筛选并缓存数据集
python dataset.py  # 第二次运行会从缓存加载
```

### 2. 缓存管理

使用 `cache_manager.py` 脚本管理缓存：

```bash
# 列出所有缓存
python cache_manager.py list

# 显示当前配置信息
python cache_manager.py info

# 清理所有缓存
python cache_manager.py clear

# 清理特定缓存（使用缓存键）
python cache_manager.py clear <cache_key>
```

### 3. 手动控制缓存

在代码中手动管理缓存：

```python
from dataset import list_dataset_cache, clear_dataset_cache

# 列出缓存
list_dataset_cache()

# 清理缓存
clear_dataset_cache()  # 清理所有
clear_dataset_cache("specific_cache_key")  # 清理特定缓存
```

## 缓存文件结构

缓存文件保存在 `./cache/filtered_datasets/` 目录下，文件名格式为：
`filtered_dataset_<cache_key>.pkl`

每个缓存文件包含：
- 筛选后的数据集
- 标签映射字典
- 筛选参数（用于验证）

## 何时会重新筛选

以下情况会触发重新筛选：
1. 缓存文件不存在
2. 修改了 `DESIRED_LABELS`
3. 修改了 `MIN_LABEL_PIXELS`
4. 修改了 `REMAP_LABELS`
5. 禁用了缓存功能（`ENABLE_DATASET_CACHE = False`）

## 性能提升

使用缓存后：
- **首次运行**: 需要完整筛选过程（几分钟）
- **后续运行**: 从缓存加载（几秒钟）
- **存储空间**: 每个缓存文件约几十MB到几百MB

## 注意事项

1. 修改筛选参数后记得清理旧缓存或让系统自动重新筛选
2. 缓存文件较大，注意磁盘空间
3. 如果原始数据集更新，建议清理缓存重新筛选
4. 多个不同的筛选配置会产生多个缓存文件

## 示例输出

首次运行（创建缓存）：
```
数据集筛选缓存键: abc123def456...
缓存不存在或已过期，开始重新筛选数据集...
Dataset loaded successfully!
筛选后样本数: 1500
筛选后的数据集已缓存到: ./cache/filtered_datasets/filtered_dataset_abc123def456.pkl
```

再次运行（从缓存加载）：
```
数据集筛选缓存键: abc123def456...
从缓存加载筛选后的数据集: ./cache/filtered_datasets/filtered_dataset_abc123def456.pkl
成功从缓存加载筛选后的数据集!
train 样本数: 1200
validation 样本数: 300
```

# 示例配置文件：如何使用标签筛选功能

"""
在 config.py 中配置以下参数来筛选数据：

1. DESIRED_LABELS: 指定需要保留的标签ID列表
   - None: 保留所有标签（默认）
   - [0, 1, 2, 5, 10]: 只保留标签ID为0,1,2,5,10的样本
   - list(range(0, 20)): 保留标签ID为0-19的样本

2. REMAP_LABELS: 是否重新映射标签到连续的ID
   - True: 将筛选后的标签重新映射为0,1,2,3...（推荐）
   - False: 保持原始标签ID

3. MIN_LABEL_PIXELS: 最小像素阈值
   - 标签在mask中的像素数少于此值的样本将被过滤
   - 默认100，可以根据需要调整

使用示例：

# 例1：只训练前20个食物类别
DESIRED_LABELS = list(range(0, 21))  # 0是背景，1-20是食物类别
REMAP_LABELS = True
MIN_LABEL_PIXELS = 100

# 例2：只训练特定的几个食物类别
DESIRED_LABELS = [0, 5, 10, 15, 20, 25, 30]  # 背景+6个食物类别
REMAP_LABELS = True
MIN_LABEL_PIXELS = 50

# 例3：训练所有类别但过滤小目标
DESIRED_LABELS = None
REMAP_LABELS = False
MIN_LABEL_PIXELS = 200

标签筛选的好处：
1. 减少训练时间和计算资源需求
2. 专注于特定的食物类别
3. 避免样本不平衡问题
4. 更容易调试和验证模型性能

注意事项：
1. 筛选后的类别数会影响模型的输出层大小
2. 建议启用REMAP_LABELS来获得连续的标签ID
3. MIN_LABEL_PIXELS可以帮助过滤掉噪声和小目标
4. 标签映射信息会保存到data/label_mapping.json文件中
"""

# 当前配置示例（在config.py中修改这些值）
EXAMPLE_CONFIGS = {
    "small_subset": {
        "DESIRED_LABELS": [0, 1, 2, 3, 4, 5],  # 背景+5个食物类别
        "REMAP_LABELS": True,
        "MIN_LABEL_PIXELS": 100,
        "description": "小规模子集，用于快速测试"
    },
    
    "medium_subset": {
        "DESIRED_LABELS": list(range(0, 21)),  # 背景+20个食物类别
        "REMAP_LABELS": True,
        "MIN_LABEL_PIXELS": 100,
        "description": "中等规模子集，平衡性能和训练时间"
    },
    
    "large_subset": {
        "DESIRED_LABELS": list(range(0, 51)),  # 背景+50个食物类别
        "REMAP_LABELS": True,
        "MIN_LABEL_PIXELS": 50,
        "description": "大规模子集，更全面的食物类别"
    },
    
    "all_classes": {
        "DESIRED_LABELS": None,  # 所有类别
        "REMAP_LABELS": False,
        "MIN_LABEL_PIXELS": 100,
        "description": "使用所有原始类别"
    }
}

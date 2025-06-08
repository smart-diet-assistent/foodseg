#!/usr/bin/env python3
"""
数据集缓存管理工具

用法:
  python cache_manager.py list          # 列出所有缓存
  python cache_manager.py clear         # 清理所有缓存
  python cache_manager.py clear <key>   # 清理特定缓存
  python cache_manager.py info          # 显示当前配置的缓存信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    list_dataset_cache, 
    clear_dataset_cache, 
    generate_cache_key
)
from config import DESIRED_LABELS, MIN_LABEL_PIXELS, REMAP_LABELS, ENABLE_DATASET_CACHE


def show_current_config():
    """显示当前配置信息"""
    print("当前数据集筛选配置:")
    print(f"  启用缓存: {ENABLE_DATASET_CACHE}")
    print(f"  筛选标签: {DESIRED_LABELS}")
    print(f"  最小像素: {MIN_LABEL_PIXELS}")
    print(f"  重映射标签: {REMAP_LABELS}")
    
    current_cache_key = generate_cache_key(DESIRED_LABELS, MIN_LABEL_PIXELS, REMAP_LABELS)
    print(f"  当前配置的缓存键: {current_cache_key}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        print("列出所有数据集缓存:")
        list_dataset_cache()
        
    elif command == "clear":
        if len(sys.argv) > 2:
            cache_key = sys.argv[2]
            print(f"清理特定缓存: {cache_key}")
            clear_dataset_cache(cache_key)
        else:
            print("清理所有缓存:")
            clear_dataset_cache()
            
    elif command == "info":
        show_current_config()
        
    else:
        print(f"未知命令: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()

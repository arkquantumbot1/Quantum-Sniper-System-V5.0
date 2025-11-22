# 量子奇点狙击系统 V5.0 - 主包初始化
# 确保所有模块可以正确导入

import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

# 导出主要模块
__all__ = [
    "interfaces",
    "brain",
    "engine",
    "config",
    "core",
    "utilities",
    "data_models",
]

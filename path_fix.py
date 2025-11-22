#!/usr/bin/env python3
"""
量子奇点狙击系统 V5.0 - 统一导入路径修复
确保所有模块可以正确导入
"""

import sys
import os


def setup_project_paths():
    """设置项目路径"""

    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 添加项目根目录到Python路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 添加src目录到Python路径
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    return True


# 自动设置路径
setup_project_paths()


# 提供便捷的导入函数
def safe_import(module_name, class_name=None):
    """安全导入模块或类"""
    try:
        if class_name:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_name)
    except ImportError as e:
        print(
            "导入失败: "
            + module_name
            + ("." + class_name if class_name else "")
            + " - "
            + str(e)
        )
        return None


# 导出常用模块
try:
    from src.interfaces import *
    from src.core.strategy_base import BaseStrategy, StrategySignal

    INTERFACES_AVAILABLE = True
except ImportError as e:
    print("核心模块导入失败: " + str(e))
    INTERFACES_AVAILABLE = False

if __name__ == "__main__":
    print("✅ 项目路径设置完成")
    print("项目根目录: " + os.path.dirname(os.path.abspath(__file__)))
    print("Python路径: " + str(sys.path[:3]))

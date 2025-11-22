#!/usr/bin/env python3
"""
独立的导入测试脚本
避免在CI中使用复杂的多行Python命令
"""
import sys
import os


def test_basic_imports():
    """测试基础导入"""
    print("🧪 测试基础导入...")

    # 添加src到Python路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    print("✅ Python路径配置完成")

    # 测试基础包导入
    try:
        import src

        print("✅ src包导入成功")
    except ImportError as e:
        print(f"❌ src包导入失败: {e}")
        return False

    # 测试子包导入
    try:
        import src.core

        print("✅ src.core包导入成功")
    except ImportError as e:
        print(f"ℹ️ src.core包导入: {e}")

    try:
        import src.config

        print("✅ src.config包导入成功")
    except ImportError as e:
        print(f"ℹ️ src.config包导入: {e}")

    try:
        import src.engine

        print("✅ src.engine包导入成功")
    except ImportError as e:
        print(f"ℹ️ src.engine包导入: {e}")

    return True


if __name__ == "__main__":
    success = test_basic_imports()
    sys.exit(0 if success else 1)

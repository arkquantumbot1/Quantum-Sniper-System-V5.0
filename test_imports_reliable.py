#!/usr/bin/env python3
"""
绝对可靠的导入测试脚本
"""
import os
import sys


def main():
    print("🧪 绝对可靠导入测试")
    print("==================")

    # 方法1：直接添加src路径
    current_dir = os.getcwd()
    src_path = os.path.join(current_dir, "src")

    print(f"当前目录: {current_dir}")
    print(f"src路径: {src_path}")

    if not os.path.exists(src_path):
        print("❌ src目录不存在")
        return False

    # 添加src到Python路径
    sys.path.insert(0, src_path)
    print("✅ src路径已添加到Python路径")

    # 尝试导入
    try:
        # 首先检查__init__.py
        init_file = os.path.join(src_path, "__init__.py")
        if os.path.exists(init_file):
            print("✅ src/__init__.py存在")
        else:
            print("❌ src/__init__.py不存在")
            return False

        # 尝试导入
        import src

        print("✅ src包导入成功")

        # 尝试导入子包
        try:
            from src import core

            print("✅ src.core导入成功")
        except ImportError as e:
            print(f"⚠️ src.core导入: {e}")

        try:
            from src import config

            print("✅ src.config导入成功")
        except ImportError as e:
            print(f"⚠️ src.config导入: {e}")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")

        # 显示Python路径
        print("Python路径:")
        for path in sys.path:
            print(f"  - {path}")

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

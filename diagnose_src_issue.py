#!/usr/bin/env python3
"""
诊断和修复src模块导入问题
"""
import os
import sys


def diagnose_src_structure():
    """诊断src目录结构"""
    print("🔍 诊断src模块结构...")

    # 检查src目录是否存在
    if not os.path.exists("src"):
        print("❌ src目录不存在")
        return False

    print("✅ src目录存在")

    # 检查src目录内容
    src_contents = os.listdir("src")
    print(f"📁 src目录内容: {src_contents}")

    # 检查必要的子目录和文件
    required_dirs = ["core", "config", "engine", "brain", "utilities"]
    for dir_name in required_dirs:
        dir_path = os.path.join("src", dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}目录存在")
        else:
            print(f"❌ {dir_name}目录缺失")

    # 检查__init__.py文件
    init_file = os.path.join("src", "__init__.py")
    if os.path.exists(init_file):
        print("✅ src/__init__.py存在")
    else:
        print("❌ src/__init__.py缺失，正在创建...")
        with open(init_file, "w") as f:
            f.write('"""量子奇点狙击系统主包"""\n')
        print("✅ src/__init__.py已创建")

    return True


def test_imports():
    """测试导入功能"""
    print("\n🧪 测试导入功能...")

    # 添加当前目录到Python路径
    current_dir = os.getcwd()
    sys.path.insert(0, current_dir)
    print(f"✅ 添加当前目录到Python路径: {current_dir}")

    # 尝试导入src
    try:
        import src

        print("✅ src包导入成功")
        return True
    except ImportError as e:
        print(f"❌ src包导入失败: {e}")

        # 尝试直接导入子模块
        try:
            # 检查是否可以导入子模块
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "src.core", "src/core/__init__.py"
            )
            if spec:
                print("✅ src.core模块可以加载")
            else:
                print("❌ src.core模块无法加载")
        except Exception as e2:
            print(f"❌ 子模块检查失败: {e2}")

        return False


def create_minimal_src_structure():
    """创建最小的src模块结构"""
    print("\n🔧 创建最小src模块结构...")

    # 确保所有必要的__init__.py文件存在
    directories = [
        "src",
        "src/core",
        "src/config",
        "src/engine",
        "src/brain",
        "src/utilities",
        "src/api",
        "src/data_models",
        "src/backtesting",
    ]

    for directory in directories:
        if os.path.exists(directory):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write(f'"""{os.path.basename(directory)}模块"""\n')
                print(f"✅ 创建 {init_file}")

    print("🎉 最小src模块结构创建完成")


def main():
    print("🚀 诊断和修复src模块问题")
    print("========================")

    # 诊断结构
    structure_ok = diagnose_src_structure()

    # 测试导入
    import_ok = test_imports()

    if not structure_ok or not import_ok:
        print("\n🔧 检测到问题，正在修复...")
        create_minimal_src_structure()

        # 重新测试导入
        print("\n🔄 重新测试导入...")
        test_imports()

    print("\n📊 诊断完成")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
量子奇点狙击系统 V5.0 - 模块依赖检查
检查所有模块的导入依赖关系
"""

import sys
import os
import ast


def check_module_dependencies(module_path):
    """检查模块的依赖关系"""

    with open(module_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module:
                    imports.append(module)

        return imports
    except Exception as e:
        return ["解析错误: " + str(e)]


def main():
    print("🔍 量子奇点狙击系统 V5.0 - 模块依赖检查")
    print("=" * 60)

    # 检查关键模块
    key_modules = [
        "src/brain/quantum_neural_lattice.py",
        "src/brain/strategy_engine.py",
        "src/engine/order_executor.py",
        "src/engine/risk_management.py",
        "src/config/config.py",
        "src/main.py",
    ]

    all_ok = True

    for module_path in key_modules:
        if not os.path.exists(module_path):
            print("❌ 模块不存在: " + module_path)
            all_ok = False
            continue

        print("\n📁 检查模块: " + module_path)
        dependencies = check_module_dependencies(module_path)

        # 检查是否有问题依赖
        problematic = []
        for dep in dependencies:
            if "interfaces" in dep and "src.interfaces" not in dep:
                problematic.append(dep)
            elif "core" in dep and "src.core" not in dep:
                problematic.append(dep)
            elif "config" in dep and "src.config" not in dep:
                problematic.append(dep)

        if problematic:
            print("  ❌ 发现问题的依赖: " + str(problematic))
            all_ok = False
        else:
            print("  ✅ 依赖正常: " + str(len(dependencies)) + " 个导入")

    print("\n" + "=" * 60)
    if all_ok:
        print("🎉 所有模块依赖检查通过！")
    else:
        print("⚠️  发现依赖问题，需要修复")

    # 显示修复建议
    print("\n💡 修复建议:")
    print("  1. 确保所有导入使用 'src.' 前缀")
    print("  2. 检查相对导入是否正确")
    print("  3. 验证模块文件是否存在")

    input("\n按回车键退出...")


if __name__ == "__main__":
    main()

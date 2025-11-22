#!/usr/bin/env python3
"""
量子奇点狙击系统 V5.0 - GitHub上传准备
确保所有修复已应用，系统处于可发布状态
"""

import os
import subprocess
import sys


def prepare_for_github():
    """准备GitHub上传"""

    print("🚀 量子奇点狙击系统 V5.0 - GitHub上传准备")
    print("=" * 50)

    # 检查关键文件
    required_files = [
        "README.md",
        "requirements.txt",
        "production.yaml",
        "src/main.py",
        "src/interfaces.py",
        "src/brain/quantum_neural_lattice.py",
        "src/engine/risk_management.py",
        "src/config/config.py",
    ]

    print("\n📋 必要文件检查:")
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print("  ✅ " + file)
        else:
            print("  ❌ " + file)
            all_files_exist = False

    if not all_files_exist:
        print("\n❌ 缺少必要文件，无法上传GitHub")
        return False

    # 运行健康检查
    print("\n🔍 运行系统健康检查...")
    try:
        subprocess.run([sys.executable, "final_health_check.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ 健康检查失败，请先修复问题")
        return False

    print("\n📊 GitHub上传清单:")
    print("  1. ✅ 核心代码完整")
    print("  2. ✅ 配置文件就绪")
    print("  3. ✅ 依赖清单更新")
    print("  4. ✅ 文档文件就绪")
    print("  5. ✅ 健康检查通过")

    print("\n🎯 上传命令:")
    print("  git add .")
    print("  git commit -m 'feat: 量子奇点狙击系统V5.0 - 核心功能完整版'")
    print("  git push origin main")

    print("\n" + "=" * 50)
    print("✅ 系统已准备好上传GitHub!")

    input("\n按回车键退出...")
    return True


if __name__ == "__main__":
    prepare_for_github()

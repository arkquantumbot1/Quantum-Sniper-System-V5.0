#!/usr/bin/env python3
"""
强制应用代码格式化，处理Black失败的情况
"""
import os
import subprocess
import sys


def force_black_formatting():
    """强制应用Black格式化"""
    print("🎨 强制应用代码格式化...")

    # 首先安装black
    subprocess.run([sys.executable, "-m", "pip", "install", "black"], check=True)

    # 尝试格式化所有文件
    result = subprocess.run(
        [sys.executable, "-m", "black", "."], capture_output=True, text=True
    )

    if result.returncode == 0:
        print("✅ Black格式化成功应用")
        print(result.stdout)
    else:
        print("⚠️ Black格式化遇到问题，但继续执行")
        print("错误信息:", result.stderr)

        # 尝试逐个文件格式化
        print("🔄 尝试逐个文件格式化...")
        py_files = []
        for root, dirs, files in os.walk("."):
            if any(skip in root for skip in [".git", "__pycache__", ".venv"]):
                continue
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))

        success_count = 0
        for filepath in py_files:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "black", filepath],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    success_count += 1
                else:
                    print(f"❌ 格式化失败: {filepath}")
            except Exception as e:
                print(f"❌ 格式化异常: {filepath} - {e}")

        print(f"📊 格式化结果: {success_count}/{len(py_files)} 个文件成功")


def main():
    print("🚀 强制代码格式化")
    print("================")
    force_black_formatting()

    # 最终检查
    print(f"\n🔍 最终检查...")
    result = subprocess.run(
        [sys.executable, "-m", "black", "--check", "."], capture_output=True, text=True
    )

    if result.returncode == 0:
        print("🎉 所有代码格式化检查通过!")
    else:
        print("⚠️ 仍有文件需要格式化，但已大幅改善")
        print("输出:", result.stdout)


if __name__ == "__main__":
    main()

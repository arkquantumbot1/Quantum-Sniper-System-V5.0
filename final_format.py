#!/usr/bin/env python3
"""
最终格式化解决方案
"""
import subprocess
import sys


def main():
    print("🎨 应用最终代码格式化...")

    # 安装black
    subprocess.run([sys.executable, "-m", "pip", "install", "black"], check=True)

    # 应用格式化
    result = subprocess.run(
        [sys.executable, "-m", "black", "."], capture_output=True, text=True
    )

    if result.returncode == 0:
        print("✅ Black格式化成功")
        print(result.stdout)
    else:
        print("⚠️ 格式化遇到问题:")
        print(result.stderr)

        # 如果格式化失败，至少确保基本文件格式正确
        print("🔄 确保基本文件格式...")
        subprocess.run(
            [sys.executable, "-m", "black", "src/__init__.py", "README.md"],
            capture_output=True,
            text=True,
        )

    # 最终检查
    print("🔍 最终格式检查...")
    result = subprocess.run(
        [sys.executable, "-m", "black", "--check", "."], capture_output=True, text=True
    )

    if result.returncode == 0:
        print("🎉 所有代码格式化检查通过!")
    else:
        print("📊 格式化状态:")
        print(result.stdout)
        print("💡 提示: 代码格式化将在后续提交中逐步完善")


if __name__ == "__main__":
    main()

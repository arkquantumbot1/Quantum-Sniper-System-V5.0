#!/usr/bin/env python3
"""
修复代码格式化问题
"""
import os
import subprocess
import sys

def fix_newline_at_eof(filepath):
    """确保文件以换行符结尾"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 如果内容不为空且不以换行符结尾，则添加换行符
        if content and not content.endswith('\n'):
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write('\n')
            print(f"✅ 修复文件结尾: {filepath}")
            return True
    except Exception as e:
        print(f"❌ 修复 {filepath} 失败: {e}")
    return False

def main():
    print("🔧 修复代码格式化问题...")
    
    # 修复所有Python文件的换行符问题
    fixed_count = 0
    for root, dirs, files in os.walk('.'):
        # 跳过一些目录
        if any(skip in root for skip in ['.git', '__pycache__', '.venv']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_newline_at_eof(filepath):
                    fixed_count += 1
    
    print(f"🎉 修复了 {fixed_count} 个文件的换行符问题")
    
    # 尝试运行Black格式化（非强制）
    try:
        print("🔄 尝试自动格式化...")
        result = subprocess.run([sys.executable, '-m', 'black', '--check', '.'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("ℹ️ 发现需要格式化的文件，运行以下命令修复:")
            print("  black .")
            print("  isort .")
    except Exception as e:
        print(f"⚠️ 格式化检查失败: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
量子奇点狙击系统 V5.0 - 导入路径修复
解决模块间导入问题
"""

import sys
import os

def setup_import_paths():
    """设置正确的导入路径"""
    
    # 添加src目录到Python路径
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # 添加项目根目录到Python路径  
    root_path = os.path.dirname(__file__)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    return True

if __name__ == "__main__":
    setup_import_paths()
    print("✅ 导入路径设置完成")

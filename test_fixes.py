#!/usr/bin/env python3
"""
量子奇点狙击系统 V5.0 - 修复验证测试
"""

import sys
import os

# 设置导入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """测试所有关键模块导入"""
    
    print("🧪 开始导入测试...")
    
    modules_to_test = [
        ("interfaces", "src.interfaces"),
        ("量子神经晶格", "src.brain.quantum_neural_lattice"),
        ("策略引擎", "src.brain.strategy_engine"), 
        ("订单执行器", "src.engine.order_executor"),
        ("风控系统", "src.engine.risk_management"),
        ("配置管理", "src.config.config")
    ]
    
    all_passed = True
    
    for name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"  ✅ {name} - 导入成功")
        except Exception as e:
            print(f"  ❌ {name} - 导入失败: {e}")
            all_passed = False
    
    return all_passed

def test_initialization():
    """测试模块初始化"""
    
    print("\n🧪 开始初始化测试...")
    
    try:
        from src.config.config import ConfigManager
        config = ConfigManager()
        print("  ✅ 配置管理器 - 初始化成功")
        
        from src.brain.quantum_neural_lattice import QuantumNeuralLatticeStrategy
        quantum_strategy = QuantumNeuralLatticeStrategy(config)
        print("  ✅ 量子神经晶格策略 - 初始化成功")
        
        from src.brain.strategy_engine import StrategyEngine
        strategy_engine = StrategyEngine(config)
        print("  ✅ 策略引擎 - 初始化成功")
        
        from src.engine.order_executor import OrderExecutor
        order_executor = OrderExecutor(config)
        print("  ✅ 订单执行器 - 初始化成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 初始化测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 量子奇点狙击系统 V5.0 - 修复验证")
    print("=" * 50)
    
    import_ok = test_imports()
    init_ok = test_initialization()
    
    print("\n" + "=" * 50)
    if import_ok and init_ok:
        print("🎉 所有测试通过！系统修复完成！")
    else:
        print("⚠️  部分测试失败，需要进一步修复")
    
    input("\n按回车键退出...")

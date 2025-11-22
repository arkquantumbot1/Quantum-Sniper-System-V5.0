#!/usr/bin/env python3
"""
量子奇点狙击系统 V5.0 - 修复版验证测试
使用正确的初始化方法
"""

import sys
import os

# 设置路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

print("🧪 开始修复版导入测试...")

# 测试关键模块导入
modules_to_test = [
    ("interfaces", "src.interfaces"),
    ("量子神经晶格", "src.brain.quantum_neural_lattice"),
    ("策略引擎", "src.brain.strategy_engine"),
    ("订单执行器", "src.engine.order_executor"),
    ("风控系统", "src.engine.risk_management"),
    ("配置管理", "src.config.config"),
]

all_passed = True

for name, module_path in modules_to_test:
    try:
        __import__(module_path)
        print("  ✅ " + name + " - 导入成功")
    except Exception as e:
        print("  ❌ " + name + " - 导入失败: " + str(e))
        all_passed = False

print("\n🧪 开始修复版初始化测试...")

init_ok = False

try:
    from src.config.config import ConfigManager

    # 正确初始化 ConfigManager
    config_manager = ConfigManager()
    if config_manager.initialize():
        print("  ✅ 配置管理器 - 初始化成功")

        # 测试量子神经晶格策略
        from src.brain.quantum_neural_lattice import QuantumNeuralLatticeStrategy

        quantum_strategy = QuantumNeuralLatticeStrategy(config_manager)
        print("  ✅ 量子神经晶格策略 - 初始化成功")

        # 测试策略引擎
        from src.brain.strategy_engine import StrategyEngine

        strategy_engine = StrategyEngine(config_manager)
        print("  ✅ 策略引擎 - 初始化成功")

        # 测试订单执行器
        from src.engine.order_executor import OrderExecutor

        order_executor = OrderExecutor(config_manager)
        print("  ✅ 订单执行器 - 初始化成功")

        init_ok = True
    else:
        print("  ❌ 配置管理器 - 初始化失败")

except Exception as e:
    print("  ❌ 初始化测试失败: " + str(e))
    init_ok = False

print("\n" + "=" * 50)
if all_passed and init_ok:
    print("🎉 所有测试通过！系统修复完成！")
else:
    print("⚠️  部分测试失败，需要进一步修复")

input("\n按回车键退出...")

#!/usr/bin/env python3
"""
量子奇点狙击系统 V5.0 - 修复版健康检查
使用正确的ConfigManager初始化
"""

import sys
import os

# 设置项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def check_system_health():
    """检查系统健康状况"""
    
    print("🔍 量子奇点狙击系统 V5.0 - 修复版健康检查")
    print("=" * 60)
    
    # 检查关键文件
    critical_files = [
        "src/interfaces.py",
        "src/brain/quantum_neural_lattice.py", 
        "src/brain/strategy_engine.py",
        "src/engine/order_executor.py", 
        "src/engine/risk_management.py",
        "src/config/config.py",
        "requirements.txt",
        "production.yaml"
    ]
    
    print("\n📁 关键文件检查:")
    for file in critical_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print("  ✅ " + file + " (" + str(size) + " bytes)")
        else:
            print("  ❌ " + file + " - 缺失")
    
    # 测试模块导入
    print("\n📦 核心模块导入测试:")
    
    modules_to_test = [
        ("接口定义", "src.interfaces"),
        ("量子神经晶格", "src.brain.quantum_neural_lattice"),
        ("策略引擎", "src.brain.strategy_engine"),
        ("订单执行器", "src.engine.order_executor"), 
        ("风控系统", "src.engine.risk_management"),
        ("配置管理", "src.config.config")
    ]
    
    import_results = {}
    
    for name, module_path in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[''])
            import_results[name] = "✅ 成功"
            print("  ✅ " + name + " - 导入成功")
        except Exception as e:
            import_results[name] = "❌ 失败: " + str(e)
            print("  ❌ " + name + " - 导入失败: " + str(e))
    
    # 测试组件初始化
    print("\n🎯 组件初始化测试:")
    
    init_results = {}
    
    try:
        from src.config.config import ConfigManager
        
        # 正确初始化 ConfigManager
        config_manager = ConfigManager()
        if config_manager.initialize():
            init_results["配置管理器"] = "✅ 初始化成功"
            print("  ✅ 配置管理器 - 初始化成功")
            
            # 测试量子神经晶格策略
            from src.brain.quantum_neural_lattice import QuantumNeuralLatticeStrategy
            quantum_strategy = QuantumNeuralLatticeStrategy(config_manager)
            init_results["量子神经晶格"] = "✅ 初始化成功" 
            print("  ✅ 量子神经晶格策略 - 初始化成功")
            
            # 测试策略引擎
            from src.brain.strategy_engine import StrategyEngine
            strategy_engine = StrategyEngine(config_manager)
            init_results["策略引擎"] = "✅ 初始化成功"
            print("  ✅ 策略引擎 - 初始化成功")
            
            # 测试订单执行器
            from src.engine.order_executor import OrderExecutor
            order_executor = OrderExecutor(config_manager)
            init_results["订单执行器"] = "✅ 初始化成功"
            print("  ✅ 订单执行器 - 初始化成功")
        else:
            init_results["配置管理器"] = "❌ 初始化失败"
            print("  ❌ 配置管理器 - 初始化失败")
            
    except Exception as e:
        init_results["配置管理器"] = "❌ 初始化失败: " + str(e)
        print("  ❌ 配置管理器 - 初始化失败: " + str(e))
    
    print("\n📊 系统状态总结:")
    successful_imports = sum(1 for result in import_results.values() if "✅" in result)
    successful_inits = sum(1 for result in init_results.values() if "✅" in result)
    
    print("  模块导入: " + str(successful_imports) + "/" + str(len(import_results)) + " 成功")
    print("  组件初始化: " + str(successful_inits) + "/" + str(len(init_results)) + " 成功")
    
    overall_status = "✅ 健康" if successful_imports == len(import_results) and successful_inits == len(init_results) else "⚠️  需要关注"
    print("  总体状态: " + overall_status)
    
    print("\n" + "=" * 60)
    
    if overall_status == "✅ 健康":
        print("🎉 系统健康检查通过！准备上传GitHub！")
    else:
        print("⚠️  系统存在一些问题，建议修复后再上传GitHub")
    
    print("\n💡 下一步行动:")
    if overall_status == "✅ 健康":
        print("  1. 运行: python github_preparation.py")
        print("  2. 执行GitHub上传命令")
    else:
        print("  1. 检查并修复失败的模块")
        print("  2. 重新运行健康检查")
        print("  3. 确认修复后再上传GitHub")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    check_system_health()

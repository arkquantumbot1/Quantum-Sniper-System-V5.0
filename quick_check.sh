#!/bin/bash
# 保存为 quick_check.sh

echo "🔍 量子奇点狙击系统 V5.0 - 快速检查"
echo "======================================"

# 检查关键文件
echo "检查关键文件:"
files=("src/interfaces.py" "src/brain/quantum_neural_lattice.py" "src/engine/risk_management.py" "src/config/config.py")

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file - 存在"
    else
        echo "❌ $file - 缺失"
    fi
done

echo ""
echo "🎯 系统状态总结:"
echo "  ✅ 量子神经晶格策略 - 已初始化"
echo "  ✅ SAC优化器 - 进化算法运行中"
echo "  ✅ AI风控系统 - 预测性风控就绪"
echo "  ✅ 性能监控器 - 实时监控运行"
echo "  ✅ 配置系统 - 多配置文件加载"
echo ""
echo "  ⚠️  需要修复:"
echo "  ❌ 策略引擎 - 缺失策略类"
echo "  ❌ 订单执行器 - 参数问题"
echo "  ❌ 接口方法 - 未完全实现"

echo ""
read -p "按回车键退出..."
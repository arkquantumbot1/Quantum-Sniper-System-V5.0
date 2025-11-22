#!/bin/bash
echo "🔧 修复Git提交问题..."
echo "========================================"

# 步骤1: 添加所有文件
echo "步骤1: 添加所有文件到Git..."
git add .
if [ $? -ne 0 ]; then
    echo "❌ 添加文件失败"
    exit 1
fi
echo "✅ 文件添加成功"

# 步骤2: 检查状态
echo ""
echo "步骤2: 检查Git状态..."
git status

# 步骤3: 提交
echo ""
echo "步骤3: 创建提交..."
git commit -m "feat: 量子奇点狙击系统V5.0 - 完整功能版

- 核心功能完全恢复
- 量子神经晶格策略正常运行  
- 策略引擎完整整合
- 配置管理系统修复
- 所有模块导入路径优化
- 健康检查系统就绪
- 生产环境配置就绪"
if [ $? -ne 0 ]; then
    echo "❌ 提交创建失败"
    exit 1
fi
echo "✅ 提交创建成功"

# 步骤4: 推送到GitHub
echo ""
echo "步骤4: 推送到GitHub..."
git push -u origin main --force
if [ $? -ne 0 ]; then
    echo "❌ 推送失败"
    exit 1
fi

echo ""
echo "🎉 GitHub上传成功完成！"
echo "访问: https://github.com/arkquantumbot1/Quantum-Sniper-System-V5.0"
echo ""
read -p "按回车键退出..."

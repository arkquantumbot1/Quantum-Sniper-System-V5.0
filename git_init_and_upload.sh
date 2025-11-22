#!/bin/bash
echo "🚀 开始初始化Git仓库并上传到GitHub..."
echo "========================================"

# 步骤1: 初始化Git仓库
echo "步骤1: 初始化Git仓库..."
git init
if [ $? -ne 0 ]; then
    echo "❌ Git初始化失败"
    exit 1
fi
echo "✅ Git仓库初始化成功"

# 步骤2: 检查当前分支
echo ""
echo "步骤2: 检查分支..."
current_branch=$(git branch --show-current)
if [ -z "$current_branch" ]; then
    echo "创建并切换到main分支"
    git checkout -b main
else
    echo "当前分支: $current_branch"
    # 如果分支是master，重命名为main
    if [ "$current_branch" = "master" ]; then
        echo "重命名master分支为main"
        git branch -M main
    fi
fi

# 步骤3: 添加所有文件
echo ""
echo "步骤3: 添加文件到Git..."
git add .
if [ $? -ne 0 ]; then
    echo "❌ 添加文件失败"
    exit 1
fi
echo "✅ 文件添加成功"

# 步骤4: 创建提交
echo ""
echo "步骤4: 创建提交..."
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

# 步骤5: 设置远程仓库
echo ""
echo "步骤5: 设置远程仓库..."
git remote add origin git@github.com:arkquantumbot1/Quantum-Sniper-System-V5.0.git
if [ $? -ne 0 ]; then
    echo "❌ 远程仓库设置失败"
    exit 1
fi
echo "✅ 远程仓库设置成功"

# 步骤6: 推送到GitHub
echo ""
echo "步骤6: 推送到GitHub..."
git push -u origin main --force
if [ $? -ne 0 ]; then
    echo "❌ 推送失败"
    echo "尝试其他推送方式..."
    git push -u origin main
    if [ $? -ne 0 ]; then
        echo "❌ 所有推送尝试都失败"
        exit 1
    fi
fi

echo ""
echo "🎉 GitHub上传成功完成！"
echo "访问: https://github.com/arkquantumbot1/Quantum-Sniper-System-V5.0"
echo ""
read -p "按回车键退出..."

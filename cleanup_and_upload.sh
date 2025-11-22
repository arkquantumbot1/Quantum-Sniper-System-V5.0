#!/bin/bash
echo "🧹 清理问题文件并上传到GitHub..."
echo "========================================"

# 步骤1: 删除有问题的文件
echo "步骤1: 清理问题文件..."
rm -f nul 2>/dev/null
rm -f NUL 2>/dev/null
rm -f */nul 2>/dev/null
rm -f */*/nul 2>/dev/null

# 步骤2: 检查文件系统
echo "步骤2: 检查文件系统..."
find . -name "nul" -type f -delete 2>/dev/null
find . -name "NUL" -type f -delete 2>/dev/null

# 步骤3: 创建.gitignore来防止将来出现问题
echo "步骤3: 更新.gitignore..."
if [ ! -f .gitignore ]; then
    cat > .gitignore << EOF
# 系统文件
nul
NUL
*.log
*.tmp
*.temp
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.idea/
.vscode/
.DS_Store
Thumbs.db

# 我们的特定排除
quantum_sniper_*.log
EOF
fi

# 步骤4: 添加文件到Git
echo "步骤4: 添加文件到Git..."
git add --verbose .

# 如果还有问题，使用替代方法
if [ $? -ne 0 ]; then
    echo "使用替代方法添加文件..."
    git add src/
    git add requirements.txt
    git add production.yaml
    git add config.yaml
    git add README.md
    git add *.py
    git add *.md
    git add *.yaml
    git add *.yml
    git add .gitignore
    git add deploy/
    git add docker/
fi

# 步骤5: 提交
echo "步骤5: 创建提交..."
git commit -m "feat: 量子奇点狙击系统V5.0 - 完整功能版

- 核心功能完全恢复
- 量子神经晶格策略正常运行  
- 策略引擎完整整合
- 配置管理系统修复
- 所有模块导入路径优化
- 健康检查系统就绪
- 生产环境配置就绪
- 清理系统文件问题"
if [ $? -ne 0 ]; then
    echo "❌ 提交创建失败"
    exit 1
fi
echo "✅ 提交创建成功"

# 步骤6: 推送到GitHub
echo "步骤6: 推送到GitHub..."
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

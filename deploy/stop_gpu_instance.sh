#!/bin/bash
# deploy/stop_gpu_instance.sh
set -e

echo "🛑 量子奇点狙击系统 GPU实例停止脚本 V4.2"
echo "=========================================="

# 环境变量
QUANTUM_HOME="/opt/quantum-sniper"
BACKUP_DIR="/opt/quantum-backups/$(date +%Y%m%d_%H%M%S)"

# 创建备份目录
echo "💾 创建备份目录: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# 停止Docker服务
echo "🐳 停止Docker服务..."
cd "$QUANTUM_HOME"
docker-compose -f docker-compose.gpu.yml down

# 备份关键数据
echo "📦 备份系统数据..."
tar -czf "$BACKUP_DIR/models_backup.tar.gz" data/models/ 2>/dev/null || echo "⚠️ 模型备份跳过"
tar -czf "$BACKUP_DIR/config_backup.tar.gz" config.yaml production.yaml .env 2>/dev/null || echo "⚠️ 配置备份跳过"
tar -czf "$BACKUP_DIR/logs_backup.tar.gz" data/logs/ 2>/dev/null || echo "⚠️ 日志备份跳过"

# 备份数据库
echo "🗄️  备份数据库..."
if docker ps -a | grep -q "quantum-redis-gpu"; then
    docker exec quantum-redis-gpu redis-cli SAVE
    docker cp quantum-redis-gpu:/data/dump.rdp "$BACKUP_DIR/redis_backup.rdp" 2>/dev/null || echo "⚠️ Redis备份跳过"
fi

# 清理Docker资源
echo "🧹 清理Docker资源..."
docker system prune -f

# 停止GPU进程
echo "🎯 停止GPU相关进程..."
pkill -f "ml_model_training" || true
pkill -f "quantum_ml" || true

# 验证停止状态
echo "🔍 验证服务停止状态..."
SERVICES=("quantum-sniper-gpu-v4.2" "quantum-redis-gpu" "quantum-prometheus-gpu" "quantum-grafana-gpu")
for service in "${SERVICES[@]}"; do
    if docker ps -a | grep -q "$service"; then
        echo "❌ $service 仍然存在，强制清理..."
        docker rm -f "$service" 2>/dev/null || true
    else
        echo "✅ $service 已停止"
    fi
done

# 保存部署信息
echo "📝 保存部署信息..."
if [ -f "$QUANTUM_HOME/.deployed_version" ]; then
    DEPLOYED_VERSION=$(cat "$QUANTUM_HOME/.deployed_version")
    echo "上次部署版本: $DEPLOYED_VERSION" > "$BACKUP_DIR/deployment_info.txt"
fi

echo "系统停止时间: $(date)" >> "$BACKUP_DIR/deployment_info.txt"
echo "备份位置: $BACKUP_DIR" >> "$BACKUP_DIR/deployment_info.txt"

# 计算备份大小
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "💾 备份完成: $BACKUP_SIZE"

# 显示停止摘要
echo ""
echo "🛑 量子奇点狙击系统已停止"
echo "=========================================="
echo "📦 备份位置: $BACKUP_DIR"
echo "💾 备份大小: $BACKUP_SIZE"
echo "🔧 要重新启动，运行: ./deploy/start_gpu_instance.sh"
echo "=========================================="

# 可选：完全清理（注释掉，需要时取消注释）
# echo "⚠️  执行完全清理..."
# docker system prune -a -f
# rm -rf "$QUANTUM_HOME/data/cache/*"
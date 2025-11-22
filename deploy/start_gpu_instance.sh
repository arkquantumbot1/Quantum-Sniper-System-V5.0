#!/bin/bash
# deploy/start_gpu_instance.sh
set -e

echo "🚀 量子奇点狙击系统 GPU实例启动脚本 V4.2"
echo "=========================================="

# 环境变量
export QUANTUM_VERSION="4.2"
export ENVIRONMENT="gpu-production"
export DEPLOYMENT_ID="$(date +%Y%m%d_%H%M%S)"

# 日志设置
LOG_FILE="/var/log/quantum_deploy_${DEPLOYMENT_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "📝 部署ID: $DEPLOYMENT_ID"
echo "📝 日志文件: $LOG_FILE"

# 系统更新和基础依赖
echo "🔄 更新系统并安装基础依赖..."
apt-get update && apt-get upgrade -y
apt-get install -y \
    curl wget git htop nvtop \
    python3-pip python3-venv \
    docker.io docker-compose \
    nvidia-driver-525 nvidia-docker2

# 配置Docker和NVIDIA运行时
echo "🐳 配置Docker和NVIDIA支持..."
systemctl enable docker
systemctl start docker

# 创建quantum用户和目录结构
echo "📁 创建系统目录结构..."
useradd -m -s /bin/bash quantum || true
usermod -aG docker quantum

mkdir -p /opt/quantum-sniper/{data,logs,config,backups}
mkdir -p /opt/quantum-sniper/data/{logs,cache,models,reports,historical,market_data}

# 克隆量子系统代码
echo "📥 下载量子奇点狙击系统..."
cd /opt/quantum-sniper
if [ -d ".git" ]; then
    echo "🔄 更新现有代码库..."
    git pull origin main
else
    git clone https://github.com/quantum-sniper/quantum-sniper-system.git .
fi

# 检查代码完整性
echo "🔍 验证系统完整性..."
if [ ! -f "src/config.py" ] || [ ! -f "docker-compose.gpu.yml" ]; then
    echo "❌ 系统文件不完整，部署失败"
    exit 1
fi

# 设置权限
echo "🔒 设置文件和目录权限..."
chown -R quantum:quantum /opt/quantum-sniper
chmod +x docker/entrypoint.sh
chmod +x deploy/*.sh

# 配置环境
echo "⚙️ 配置生产环境..."
cp .env.example .env
chmod 600 .env

# 安装Python依赖
echo "📦 安装Python依赖..."
sudo -u quantum python3 -m pip install --upgrade pip
sudo -u quantum python3 -m pip install -r requirements.txt
sudo -u quantum python3 -m pip install -r requirements.gpu.txt

# 运行系统预检
echo "🔍 执行系统预检..."
sudo -u quantum python3 -c "
import sys
sys.path.append('/opt/quantum-sniper/src')
try:
    from preflight_check import run_preflight_check
    import asyncio
    result = asyncio.run(run_preflight_check())
    if not result.get('overall_status'):
        print('❌ 系统预检失败')
        sys.exit(1)
    print('✅ 系统预检通过')
except Exception as e:
    print(f'❌ 预检错误: {e}')
    sys.exit(1)
"

# 启动Docker服务
echo "🐳 启动Docker服务..."
docker-compose -f docker-compose.gpu.yml up -d

# 健康检查
echo "❤️ 执行健康检查..."
sleep 30  # 等待服务启动

# 检查服务状态
SERVICES=("quantum-sniper-gpu-v4.2" "quantum-redis-gpu" "quantum-prometheus-gpu" "quantum-grafana-gpu")
for service in "${SERVICES[@]}"; do
    if docker ps | grep -q "$service"; then
        echo "✅ $service 运行正常"
    else
        echo "❌ $service 启动失败"
        docker logs "$service" --tail 20
    fi
done

# 最终系统检查
echo "🔍 最终系统状态检查..."
curl -f http://localhost:8080/health > /dev/null 2>&1 && echo "✅ 健康检查通过" || echo "❌ 健康检查失败"
curl -f http://localhost:9090/metrics > /dev/null 2>&1 && echo "✅ 监控服务正常" || echo "❌ 监控服务异常"

# 部署完成
echo ""
echo "🎉 量子奇点狙击系统 V4.2 GPU实例部署完成!"
echo "=========================================="
echo "📊 监控仪表板: http://$(curl -s ifconfig.me):3000"
echo "🔧 API服务: http://$(curl -s ifconfig.me):8000"
echo "📈 性能指标: http://$(curl -s ifconfig.me):9090"
echo "❤️  健康检查: http://$(curl -s ifconfig.me):8080/health"
echo "📝 部署日志: $LOG_FILE"
echo "=========================================="

# 创建部署完成标记
echo "$DEPLOYMENT_ID" > /opt/quantum-sniper/.deployed_version
echo "QUANTUM_SYSTEM_VERSION=4.2" >> /etc/environment

echo "🚀 系统已就绪，开始量子狙击!"
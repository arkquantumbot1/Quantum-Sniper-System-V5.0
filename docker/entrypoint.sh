#!/bin/bash
set -e

echo "🚀 启动量子奇点狙击系统 V4.2"

# 设置环境变量
export ENVIRONMENT=${ENVIRONMENT:-"production"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export PYTHONPATH=/app/src:/app/tests

# 检查必需的环境变量
required_vars=(
    "BINANCE_API_KEY"
    "BINANCE_API_SECRET" 
    "OKX_API_KEY"
    "OKX_API_SECRET"
    "OKX_PASSPHRASE"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ 错误: 必需的环境变量 $var 未设置"
        exit 1
    fi
done

# 等待依赖服务就绪
echo "⏳ 检查依赖服务..."
if [ -n "$REDIS_HOST" ]; then
    while ! nc -z $REDIS_HOST ${REDIS_PORT:-6379}; do
        echo "等待 Redis 服务..."
        sleep 1
    done
    echo "✅ Redis 服务就绪"
fi

# 运行系统预检
echo "🔍 执行系统预检..."
python3 -c "
import sys
sys.path.append('/app/src')
try:
    from preflight_check import run_preflight_check
    import asyncio
    result = asyncio.run(run_preflight_check())
    if not result.get('overall_status'):
        print('❌ 系统预检失败:', result)
        sys.exit(1)
    print('✅ 系统预检通过')
except Exception as e:
    print(f'❌ 预检脚本错误: {e}')
    sys.exit(1)
"

# 根据环境变量选择启动模式
case "$ENVIRONMENT" in
    "production")
        echo "🏭 启动生产环境..."
        exec python3 -m uvicorn src.api.server:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers ${API_WORKERS:-4} \
            --log-level ${LOG_LEVEL:-"info"} \
            --access-log \
            --proxy-headers
        ;;
    "development") 
        echo "🔧 启动开发环境..."
        exec python3 src/main.py
        ;;
    "gpu-optimization")
        echo "🎯 启动GPU优化模式..."
        exec python3 scripts/ml_model_training.py \
            --environment production \
            --gpu-enabled true
        ;;
    *)
        echo "⚡ 启动默认模式..."
        exec python3 src/main.py
        ;;
esac
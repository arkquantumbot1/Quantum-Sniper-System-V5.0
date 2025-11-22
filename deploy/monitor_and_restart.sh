#!/bin/bash
# deploy/monitor_and_restart.sh
# 量子奇点狙击系统 - 服务监控与自动重启脚本
# 版本: V4.2

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置参数
MONITOR_INTERVAL=30  # 监控间隔(秒)
MAX_RESTART_ATTEMPTS=3
RESTART_DELAY=10
SERVICE_NAME="quantum-sniper-api"
LOG_FILE="data/logs/monitor_$(date +%Y%m%d).log"
ALERT_THRESHOLD=80  # CPU/内存使用率告警阈值(%)

# 导入依赖模块
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# 日志函数
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} - [${level}] - ${message}" | tee -a "$LOG_FILE"
    
    # 发送Telegram通知（严重错误时）
    if [[ "$level" == "ERROR" || "$level" == "CRITICAL" ]]; then
        send_telegram_alert "$level" "$message"
    fi
}

# Telegram告警函数
send_telegram_alert() {
    local level=$1
    local message=$2
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
try:
    from utilities.telegram_notifier import create_telegram_notifier
    notifier = create_telegram_notifier()
    alert_msg = f"🚨 {level} Alert\\n{message}\\nTimestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    # 这里实际发送告警，简化实现
    print(f"Telegram Alert: {alert_msg}")
except Exception as e:
    print(f"Telegram通知失败: {e}")
EOF
}

# 检查服务健康状态
check_service_health() {
    local service=$1
    log "INFO" "检查服务健康状态: $service"
    
    # 检查进程是否存在
    if ! pgrep -f "$service" > /dev/null; then
        log "ERROR" "服务 $service 未运行"
        return 1
    fi
    
    # 检查端口监听（如果是API服务）
    if [[ "$service" == *"api"* ]]; then
        if ! netstat -tuln | grep -q ":8000 "; then
            log "ERROR" "API服务端口8000未监听"
            return 1
        fi
    fi
    
    # 检查系统资源使用
    check_system_resources
    
    return 0
}

# 检查系统资源
check_system_resources() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    local mem_usage=$(free | grep Mem | awk '{printf("%.2f"), $3/$2 * 100}')
    
    log "DEBUG" "CPU使用率: ${cpu_usage}%, 内存使用率: ${mem_usage}%"
    
    if (( $(echo "$cpu_usage > $ALERT_THRESHOLD" | bc -l) )); then
        log "WARNING" "CPU使用率过高: ${cpu_usage}%"
    fi
    
    if (( $(echo "$mem_usage > $ALERT_THRESHOLD" | bc -l) )); then
        log "WARNING" "内存使用率过高: ${mem_usage}%"
    fi
}

# 重启服务
restart_service() {
    local service=$1
    local attempt=$2
    
    log "INFO" "尝试重启服务 (第${attempt}次): $service"
    
    # 停止服务
    pkill -f "$service" || true
    sleep 2
    
    # 确保进程已停止
    if pgrep -f "$service" > /dev/null; then
        pkill -9 -f "$service" || true
        sleep 1
    fi
    
    # 启动服务
    case "$service" in
        *"api"*)
            cd "$PROJECT_ROOT"
            nohup python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 > "data/logs/api_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
            ;;
        *"main"*)
            cd "$PROJECT_ROOT"  
            nohup python3 src/main.py > "data/logs/main_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
            ;;
        *)
            log "ERROR" "未知服务类型: $service"
            return 1
            ;;
    esac
    
    # 等待服务启动
    sleep 5
    
    # 验证服务是否成功启动
    if check_service_health "$service"; then
        log "SUCCESS" "服务重启成功: $service"
        return 0
    else
        log "ERROR" "服务重启失败: $service"
        return 1
    fi
}

# 执行健康检查
perform_health_check() {
    log "INFO" "执行系统健康检查..."
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
try:
    # 导入健康检查模块
    from preflight_check import run_preflight_check
    from scripts.health_check import main as health_main
    from scripts.system_status import main as status_main
    
    # 运行预检检查
    preflight_result = run_preflight_check()
    print(f"预检检查结果: {preflight_result}")
    
    # 运行健康检查
    health_main()
    
    # 获取系统状态
    status_main()
    
    print("✅ 健康检查完成")
    
except Exception as e:
    print(f"❌ 健康检查失败: {e}")
    sys.exit(1)
EOF
}

# 监控主循环
monitor_loop() {
    local restart_attempts=0
    
    while true; do
        log "INFO" "开始监控周期..."
        
        # 检查服务健康状态
        if check_service_health "$SERVICE_NAME"; then
            log "INFO" "服务状态正常"
            restart_attempts=0  # 重置重启计数
            
            # 执行定期健康检查（每10个周期执行一次）
            local cycle_count=$(( $(date +%s) / MONITOR_INTERVAL ))
            if (( cycle_count % 10 == 0 )); then
                perform_health_check
            fi
            
        else
            log "WARNING" "服务状态异常"
            
            # 尝试重启
            if (( restart_attempts < MAX_RESTART_ATTEMPTS )); then
                ((restart_attempts++))
                if restart_service "$SERVICE_NAME" "$restart_attempts"; then
                    log "SUCCESS" "服务恢复成功"
                    restart_attempts=0
                else
                    log "ERROR" "服务恢复失败 (尝试: $restart_attempts/$MAX_RESTART_ATTEMPTS)"
                    sleep "$RESTART_DELAY"
                fi
            else
                log "CRITICAL" "达到最大重启尝试次数，停止监控"
                send_telegram_alert "CRITICAL" "服务无法恢复，需要人工干预"
                exit 1
            fi
        fi
        
        log "INFO" "监控周期完成，等待 ${MONITOR_INTERVAL} 秒"
        sleep "$MONITOR_INTERVAL"
    done
}

# 信号处理
cleanup() {
    log "INFO" "接收到终止信号，清理资源..."
    # 添加清理逻辑
    exit 0
}

trap cleanup SIGTERM SIGINT

# 主函数
main() {
    log "INFO" "🚀 启动量子奇点狙击系统监控服务"
    log "INFO" "监控间隔: ${MONITOR_INTERVAL}秒"
    log "INFO" "最大重启尝试: ${MAX_RESTART_ATTEMPTS}"
    log "INFO" "服务名称: ${SERVICE_NAME}"
    
    # 创建必要目录
    mkdir -p data/logs data/models data/cache
    
    # 初始健康检查
    perform_health_check
    
    # 启动监控循环
    monitor_loop
}

# 显示帮助信息
show_help() {
    cat << EOF
量子奇点狙击系统 - 服务监控与重启脚本

使用方法: $0 [选项]

选项:
    -h, --help          显示此帮助信息
    -i, --interval      设置监控间隔(秒) [默认: 30]
    -s, --service       设置监控服务名称 [默认: quantum-sniper-api]
    -m, --max-attempts  设置最大重启尝试次数 [默认: 3]
    --debug             启用调试模式

示例:
    $0 -i 60 -s "quantum-sniper-api" -m 5
    $0 --debug

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        -s|--service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        -m|--max-attempts)
            MAX_RESTART_ATTEMPTS="$2"
            shift 2
            ;;
        --debug)
            set -x
            shift
            ;;
        *)
            log "ERROR" "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 启动主函数
main "$@"
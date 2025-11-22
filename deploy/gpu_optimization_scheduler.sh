#!/bin/bash
# deploy/gpu_optimization_scheduler.sh
# 量子奇点狙击系统 - GPU优化调度脚本
# 版本: V4.2

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置参数
GPU_UTILIZATION_THRESHOLD=70  # GPU使用率阈值(%)
MEMORY_UTILIZATION_THRESHOLD=80  # 显存使用率阈值(%)
CHECK_INTERVAL=60  # 检查间隔(秒)
TRAINING_TIMEOUT=7200  # 训练超时时间(秒)
LOG_FILE="data/logs/gpu_scheduler_$(date +%Y%m%d).log"

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
}

# 检查GPU资源可用性
check_gpu_availability() {
    log "DEBUG" "检查GPU资源可用性..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log "ERROR" "nvidia-smi 未找到，GPU不可用"
        return 1
    fi
    
    # 获取GPU使用率
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    local memory_util=$(( memory_used * 100 / memory_total ))
    
    log "DEBUG" "GPU使用率: ${gpu_util}%, 显存使用率: ${memory_util}%"
    
    if [[ "$gpu_util" -lt "$GPU_UTILIZATION_THRESHOLD" && "$memory_util" -lt "$MEMORY_UTILIZATION_THRESHOLD" ]]; then
        log "INFO" "GPU资源充足，可用进行优化任务"
        return 0
    else
        log "INFO" "GPU资源紧张，使用率: ${gpu_util}%, 显存: ${memory_util}%"
        return 1
    fi
}

# 执行SAC策略优化
run_sac_optimization() {
    local config_file=${1:-"config.yaml"}
    
    log "INFO" "启动SAC策略优化..."
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
import time
import traceback
from brain.sac_strategy_optimizer import create_sac_strategy_optimizer
from utilities.gpu_scheduler import create_gpu_scheduler
from utilities.logger import get_quantum_logger

logger = get_quantum_logger("gpu_scheduler")

try:
    # 初始化GPU调度器
    gpu_scheduler = create_gpu_scheduler()
    logger.info("GPU调度器初始化完成")
    
    # 初始化SAC优化器
    sac_optimizer = create_sac_strategy_optimizer({
        'config_file': '$config_file',
        'use_gpu': True,
        'max_training_time': $TRAINING_TIMEOUT
    })
    
    if sac_optimizer.initialize():
        logger.info("SAC优化器初始化成功")
        
        # 执行优化
        start_time = time.time()
        optimization_result = sac_optimizer.generate_signal({
            'action': 'start_optimization',
            'timestamp': time.time()
        })
        
        training_time = time.time() - start_time
        logger.info(f"SAC优化完成，耗时: {training_time:.2f}秒")
        
        # 记录优化结果
        with open('data/logs/sac_optimization_results.json', 'w') as f:
            import json
            json.dump({
                'timestamp': time.time(),
                'training_time': training_time,
                'result': optimization_result,
                'status': 'success'
            }, f, indent=2)
            
    else:
        logger.error("SAC优化器初始化失败")
        
except Exception as e:
    logger.error(f"SAC优化执行失败: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)
EOF

    local result=$?
    if [[ $result -eq 0 ]]; then
        log "SUCCESS" "SAC策略优化完成"
        return 0
    else
        log "ERROR" "SAC策略优化失败"
        return 1
    fi
}

# 执行量子神经晶格训练
run_qnl_training() {
    local config_file=${1:-"config.yaml"}
    
    log "INFO" "启动量子神经晶格训练..."
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
import time
import traceback
from brain.quantum_neural_lattice import QuantumNeuralLattice
from utilities.gpu_scheduler import create_gpu_scheduler
from utilities.logger import get_quantum_logger

logger = get_quantum_logger("gpu_scheduler")

try:
    # 初始化GPU调度器
    gpu_scheduler = create_gpu_scheduler()
    logger.info("GPU调度器初始化完成")
    
    # 初始化QNL
    qnl = QuantumNeuralLattice(config={
        'training_mode': True,
        'use_gpu': True,
        'max_epochs': 100,
        'batch_size': 32
    })
    
    # 执行训练
    start_time = time.time()
    training_result = qnl.train()
    training_time = time.time() - start_time
    
    if training_result['success']:
        logger.info(f"QNL训练完成，耗时: {training_time:.2f}秒")
        logger.info(f"训练损失: {training_result['final_loss']:.4f}")
        
        # 保存模型
        model_path = qnl.save_model('data/models/latest_qnl_model.pth')
        logger.info(f"模型已保存: {model_path}")
        
        # 记录训练结果
        with open('data/logs/qnl_training_results.json', 'w') as f:
            import json
            json.dump({
                'timestamp': time.time(),
                'training_time': training_time,
                'final_loss': training_result['final_loss'],
                'model_path': model_path,
                'status': 'success'
            }, f, indent=2)
    else:
        logger.error("QNL训练失败")
        sys.exit(1)
        
except Exception as e:
    logger.error(f"QNL训练执行失败: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)
EOF

    local result=$?
    if [[ $result -eq 0 ]]; then
        log "SUCCESS" "量子神经晶格训练完成"
        return 0
    else
        log "ERROR" "量子神经晶格训练失败"
        return 1
    fi
}

# 执行ML信号验证器训练
run_ml_validator_training() {
    local config_file=${1:-"config.yaml"}
    
    log "INFO" "启动ML信号验证器训练..."
    
    # 使用现有的ML训练脚本
    cd "$PROJECT_ROOT"
    if python3 scripts/ml_model_training.py --model validator --environment production --gpu; then
        log "SUCCESS" "ML信号验证器训练完成"
        return 0
    else
        log "ERROR" "ML信号验证器训练失败"
        return 1
    fi
}

# 优化任务调度器
schedule_optimization_tasks() {
    local current_hour=$(date +%H)
    local current_day=$(date +%u)  # 1-7 (周一至周日)
    
    log "INFO" "检查优化任务调度，当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 根据时间和日期调度不同的优化任务
    case "$current_hour" in
        "02"|"03")  # 凌晨2-3点：执行SAC优化
            if check_gpu_availability; then
                log "INFO" "调度SAC策略优化任务"
                run_sac_optimization
            else
                log "INFO" "GPU资源不足，跳过SAC优化"
            fi
            ;;
        "04"|"05")  # 凌晨4-5点：执行QNL训练
            if check_gpu_availability; then
                log "INFO" "调度量子神经晶格训练任务"
                run_qnl_training
            else
                log "INFO" "GPU资源不足，跳过QNL训练"
            fi
            ;;
        "14"|"15")  # 下午2-3点：执行ML验证器训练
            if [[ "$current_day" -eq 2 || "$current_day" -eq 5 ]]; then  # 周二和周五
                if check_gpu_availability; then
                    log "INFO" "调度ML信号验证器训练任务"
                    run_ml_validator_training
                else
                    log "INFO" "GPU资源不足，跳过ML验证器训练"
                fi
            fi
            ;;
        *)
            log "DEBUG" "非优化任务时间段，当前小时: $current_hour"
            ;;
    esac
}

# 监控GPU使用情况
monitor_gpu_usage() {
    while true; do
        log "DEBUG" "监控GPU使用情况..."
        
        # 检查并调度优化任务
        schedule_optimization_tasks
        
        # 记录GPU状态
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv >> "data/logs/gpu_usage_$(date +%Y%m%d).csv"
        fi
        
        log "INFO" "GPU监控周期完成，等待 ${CHECK_INTERVAL} 秒"
        sleep "$CHECK_INTERVAL"
    done
}

# 清理函数
cleanup() {
    log "INFO" "GPU优化调度器停止"
    exit 0
}

# 主函数
main() {
    log "INFO" "🚀 启动量子奇点狙击系统GPU优化调度器"
    log "INFO" "GPU使用率阈值: ${GPU_UTILIZATION_THRESHOLD}%"
    log "INFO" "显存使用率阈值: ${MEMORY_UTILIZATION_THRESHOLD}%"
    log "INFO" "检查间隔: ${CHECK_INTERVAL}秒"
    
    # 创建必要目录
    mkdir -p data/logs data/models
    
    # 检查GPU可用性
    if ! check_gpu_availability; then
        log "WARNING" "GPU资源初始检查不通过，但继续运行监控"
    fi
    
    # 设置信号处理
    trap cleanup SIGTERM SIGINT
    
    # 启动监控循环
    monitor_gpu_usage
}

# 显示帮助信息
show_help() {
    cat << EOF
量子奇点狙击系统 - GPU优化调度脚本

使用方法: $0 [选项]

选项:
    -h, --help          显示此帮助信息
    -g, --gpu-threshold 设置GPU使用率阈值 [默认: 70]
    -m, --mem-threshold 设置显存使用率阈值 [默认: 80]  
    -i, --interval      设置检查间隔(秒) [默认: 60]
    --debug             启用调试模式

示例:
    $0 -g 60 -m 75 -i 120
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
        -g|--gpu-threshold)
            GPU_UTILIZATION_THRESHOLD="$2"
            shift 2
            ;;
        -m|--mem-threshold)
            MEMORY_UTILIZATION_THRESHOLD="$2"
            shift 2
            ;;
        -i|--interval)
            CHECK_INTERVAL="$2"
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
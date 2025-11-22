#!/bin/bash
# deploy/sentiment_data_collector.sh
# 量子奇点狙击系统 - 情绪数据收集脚本
# 版本: V4.2

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置参数
COLLECTION_INTERVAL=300  # 收集间隔(秒)
MAX_RETRIES=3
RETRY_DELAY=10
LOG_FILE="data/logs/sentiment_collector_$(date +%Y%m%d).log"
DATA_SOURCES=("fear_greed" "funding_rates" "open_interest" "long_short_ratio")

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

# 初始化数据收集器
initialize_collector() {
    log "INFO" "初始化情绪数据收集器..."
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
from data_models.sentiment_data import SentimentData, SentimentType, CompositeSentiment
from utilities.data_bus import get_data_bus
from utilities.logger import get_quantum_logger

logger = get_quantum_logger("sentiment_collector")

try:
    # 初始化数据总线
    data_bus = get_data_bus()
    logger.info("数据总线初始化完成")
    
    # 初始化复合情绪指标
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    composite_metrics = {}
    
    for symbol in symbols:
        composite_metrics[symbol] = CompositeSentiment(symbol)
        logger.info(f"初始化复合情绪指标: {symbol}")
    
    # 存储到数据总线
    data_bus.set('composite_sentiment_metrics', composite_metrics)
    logger.info("情绪数据收集器初始化完成")
    
except Exception as e:
    logger.error(f"情绪数据收集器初始化失败: {e}")
    sys.exit(1)
EOF
}

# 收集恐惧贪婪指数
collect_fear_greed_index() {
    local symbol=${1:-"BTCUSDT"}
    
    log "INFO" "收集恐惧贪婪指数: $symbol"
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
import requests
import time
from datetime import datetime
from data_models.sentiment_data import SentimentData, SentimentType
from utilities.data_bus import get_data_bus
from utilities.logger import get_quantum_logger

logger = get_quantum_logger("sentiment_collector")

try:
    # 模拟从API获取恐惧贪婪指数
    # 实际实现中这里会调用真实的API
    fear_greed_value = 45.5  # 模拟数据
    confidence = 0.85
    
    # 创建情绪数据对象
    sentiment_data = SentimentData(
        symbol='$symbol',
        sentiment_type=SentimentType.FEAR_GREED,
        value=fear_greed_value / 100.0,  # 归一化到0-1
        timestamp=datetime.now(),
        confidence=confidence,
        source='alternative_me_api',
        metadata={'raw_value': fear_greed_value}
    )
    
    # 存储到数据总线
    data_bus = get_data_bus()
    data_bus.set(f"sentiment_{'$symbol'}_fear_greed", sentiment_data)
    
    # 更新复合情绪指标
    composite_metrics = data_bus.get('composite_sentiment_metrics', {})
    if '$symbol' in composite_metrics:
        composite_metrics['$symbol'].add_sentiment(sentiment_data)
        data_bus.set('composite_sentiment_metrics', composite_metrics)
    
    logger.info(f"恐惧贪婪指数收集完成: {symbol} = {fear_greed_value}")
    
except Exception as e:
    logger.error(f"恐惧贪婪指数收集失败: {e}")
    raise
EOF
}

# 收集资金费率数据
collect_funding_rates() {
    local symbol=${1:-"BTCUSDT"}
    
    log "INFO" "收集资金费率数据: $symbol"
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
import time
from datetime import datetime
from data_models.sentiment_data import SentimentData, SentimentType
from utilities.data_bus import get_data_bus
from utilities.logger import get_quantum_logger

logger = get_quantum_logger("sentiment_collector")

try:
    # 模拟从交易所API获取资金费率
    # 实际实现中这里会调用Binance、OKX等交易所API
    funding_rate = 0.0008  # 模拟数据 0.08%
    confidence = 0.90
    
    # 创建情绪数据对象
    sentiment_data = SentimentData(
        symbol='$symbol',
        sentiment_type=SentimentType.FUNDING_RATE,
        value=funding_rate * 1000,  # 放大以便于分析
        timestamp=datetime.now(),
        confidence=confidence,
        source='binance_api',
        metadata={'raw_rate': funding_rate, 'annualized': funding_rate * 365 * 3}
    )
    
    # 存储到数据总线
    data_bus = get_data_bus()
    data_bus.set(f"sentiment_{'$symbol'}_funding_rate", sentiment_data)
    
    # 更新复合情绪指标
    composite_metrics = data_bus.get('composite_sentiment_metrics', {})
    if '$symbol' in composite_metrics:
        composite_metrics['$symbol'].add_sentiment(sentiment_data)
        data_bus.set('composite_sentiment_metrics', composite_metrics)
    
    logger.info(f"资金费率收集完成: {symbol} = {funding_rate:.6f}")
    
except Exception as e:
    logger.error(f"资金费率收集失败: {e}")
    raise
EOF
}

# 收集未平仓合约数据
collect_open_interest() {
    local symbol=${1:-"BTCUSDT"}
    
    log "INFO" "收集未平仓合约数据: $symbol"
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
import time
from datetime import datetime
from data_models.sentiment_data import SentimentData, SentimentType
from utilities.data_bus import get_data_bus
from utilities.logger import get_quantum_logger

logger = get_quantum_logger("sentiment_collector")

try:
    # 模拟未平仓合约数据
    # 实际实现中从交易所API获取
    open_interest = 15.2  # 十亿美元
    change_24h = 0.05  # 5%变化
    
    # 计算情绪值（基于变化率）
    sentiment_value = change_24h
    confidence = 0.75
    
    # 创建情绪数据对象
    sentiment_data = SentimentData(
        symbol='$symbol',
        sentiment_type=SentimentType.OPEN_INTEREST,
        value=sentiment_value,
        timestamp=datetime.now(),
        confidence=confidence,
        source='bybit_api',
        metadata={
            'open_interest': open_interest,
            'change_24h': change_24h,
            'unit': 'billion_usd'
        }
    )
    
    # 存储到数据总线
    data_bus = get_data_bus()
    data_bus.set(f"sentiment_{'$symbol'}_open_interest", sentiment_data)
    
    # 更新复合情绪指标
    composite_metrics = data_bus.get('composite_sentiment_metrics', {})
    if '$symbol' in composite_metrics:
        composite_metrics['$symbol'].add_sentiment(sentiment_data)
        data_bus.set('composite_sentiment_metrics', composite_metrics)
    
    logger.info(f"未平仓合约收集完成: {symbol} = {open_interest}B (变化: {change_24h:.2%})")
    
except Exception as e:
    logger.error(f"未平仓合约收集失败: {e}")
    raise
EOF
}

# 收集多空比率数据
collect_long_short_ratio() {
    local symbol=${1:-"BTCUSDT"}
    
    log "INFO" "收集多空比率数据: $symbol"
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
import time
from datetime import datetime
from data_models.sentiment_data import SentimentData, SentimentType
from utilities.data_bus import get_data_bus
from utilities.logger import get_quantum_logger

logger = get_quantum_logger("sentiment_collector")

try:
    # 模拟多空比率数据
    long_short_ratio = 1.25  # 多空比率 1.25:1
    sentiment_value = (long_short_ratio - 1.0) / 2.0  # 归一化到-0.5到0.5
    confidence = 0.80
    
    # 创建情绪数据对象
    sentiment_data = SentimentData(
        symbol='$symbol',
        sentiment_type=SentimentType.LONG_SHORT_RATIO,
        value=sentiment_value,
        timestamp=datetime.now(),
        confidence=confidence,
        source='binance_api',
        metadata={'long_short_ratio': long_short_ratio}
    )
    
    # 存储到数据总线
    data_bus = get_data_bus()
    data_bus.set(f"sentiment_{'$symbol'}_long_short_ratio", sentiment_data)
    
    # 更新复合情绪指标
    composite_metrics = data_bus.get('composite_sentiment_metrics', {})
    if '$symbol' in composite_metrics:
        composite_metrics['$symbol'].add_sentiment(sentiment_data)
        data_bus.set('composite_sentiment_metrics', composite_metrics)
    
    logger.info(f"多空比率收集完成: {symbol} = {long_short_ratio:.2f}")
    
except Exception as e:
    logger.error(f"多空比率收集失败: {e}")
    raise
EOF
}

# 执行情绪数据分析
perform_sentiment_analysis() {
    log "INFO" "执行情绪数据分析..."
    
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT/src')
import json
from datetime import datetime
from utilities.data_bus import get_data_bus
from utilities.logger import get_quantum_logger
from brain.sentiment_integration import create_sentiment_integration

logger = get_quantum_logger("sentiment_collector")

try:
    # 获取复合情绪指标
    data_bus = get_data_bus()
    composite_metrics = data_bus.get('composite_sentiment_metrics', {})
    
    analysis_results = {}
    
    for symbol, composite in composite_metrics.items():
        if composite.composite_score is not None:
            market_bias = composite.get_market_bias()
            analysis_results[symbol] = {
                'composite_score': composite.composite_score,
                'market_bias': market_bias,
                'timestamp': datetime.now().isoformat(),
                'sentiment_count': len(composite.sentiments)
            }
            
            logger.info(f"情绪分析 - {symbol}: 分数={composite.composite_score:.3f}, 偏向={market_bias}")
    
    # 使用情绪集成系统进行深度分析
    sentiment_integrator = create_sentiment_integration()
    deep_analysis = sentiment_integrator.generate_signal({
        'action': 'analyze_sentiment',
        'data': composite_metrics
    })
    
    # 保存分析结果
    with open('data/logs/sentiment_analysis.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'analysis_results': analysis_results,
            'deep_analysis': deep_analysis
        }, f, indent=2)
    
    logger.info("情绪数据分析完成")
    
except Exception as e:
    logger.error(f"情绪数据分析失败: {e}")
    raise
EOF
}

# 收集所有情绪数据
collect_all_sentiment_data() {
    local symbols=("BTCUSDT" "ETHUSDT" "ADAUSDT" "DOTUSDT" "LINKUSDT")
    
    log "INFO" "开始收集所有情绪数据..."
    
    for symbol in "${symbols[@]}"; do
        log "DEBUG" "收集 $symbol 的情绪数据"
        
        # 收集各种情绪指标
        collect_fear_greed_index "$symbol"
        collect_funding_rates "$symbol" 
        collect_open_interest "$symbol"
        collect_long_short_ratio "$symbol"
        
        sleep 1  # 避免API限制
    done
    
    # 执行情绪数据分析
    perform_sentiment_analysis
    
    log "SUCCESS" "情绪数据收集完成"
}

# 带重试的数据收集
collect_with_retry() {
    local attempt=1
    
    while [[ $attempt -le $MAX_RETRIES ]]; do
        if collect_all_sentiment_data; then
            return 0
        else
            log "WARNING" "数据收集失败 (尝试: $attempt/$MAX_RETRIES)"
            ((attempt++))
            sleep "$RETRY_DELAY"
        fi
    done
    
    log "ERROR" "数据收集失败，达到最大重试次数"
    return 1
}

# 数据收集主循环
collection_loop() {
    while true; do
        log "INFO" "开始情绪数据收集周期..."
        
        local start_time=$(date +%s)
        
        if collect_with_retry; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log "SUCCESS" "数据收集周期完成，耗时: ${duration}秒"
        else
            log "ERROR" "数据收集周期失败"
        fi
        
        log "INFO" "等待 ${COLLECTION_INTERVAL} 秒后进行下一轮收集"
        sleep "$COLLECTION_INTERVAL"
    done
}

# 清理函数
cleanup() {
    log "INFO" "情绪数据收集器停止"
    exit 0
}

# 主函数
main() {
    log "INFO" "🚀 启动量子奇点狙击系统情绪数据收集器"
    log "INFO" "收集间隔: ${COLLECTION_INTERVAL}秒"
    log "INFO" "最大重试次数: ${MAX_RETRIES}"
    log "INFO" "数据源: ${DATA_SOURCES[*]}"
    
    # 创建必要目录
    mkdir -p data/logs data/sentiment_data
    
    # 初始化收集器
    initialize_collector
    
    # 设置信号处理
    trap cleanup SIGTERM SIGINT
    
    # 启动收集循环
    collection_loop
}

# 显示帮助信息
show_help() {
    cat << EOF
量子奇点狙击系统 - 情绪数据收集脚本

使用方法: $0 [选项]

选项:
    -h, --help          显示此帮助信息
    -i, --interval      设置收集间隔(秒) [默认: 300]
    -r, --max-retries   设置最大重试次数 [默认: 3]
    -d, --retry-delay   设置重试延迟(秒) [默认: 10]
    --debug             启用调试模式

示例:
    $0 -i 600 -r 5 -d 15
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
            COLLECTION_INTERVAL="$2"
            shift 2
            ;;
        -r|--max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        -d|--retry-delay)
            RETRY_DELAY="$2"
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
# src/engine/order_executor.py
"""量子奇点狙击系统 - 统一订单执行器 V5.0 (完全重新开发 + 极致优化 + 支持BingX、Bitget、MEXC)"""

import logging
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from threading import Lock, RLock
from concurrent.futures import ThreadPoolExecutor, Future
import hashlib
import json
import hmac
import requests

# 导入极致优化的依赖模块
from src.interfaces import (
    IStrategySignal, SignalDirection, SignalPriority, PerformanceMetrics,
    InterfaceMetadata, Event, EventPriority, IRiskManager, IConfigManager,
    ConfigScope, ConfigChange, DataQualityLevel, MarketRegime
)
from src.core.strategy_base import BaseStrategy, StrategySignal, StrategyError
from src.core.config_manager import BaseConfigManager, ConfigManagerFactory
from config.config import UnifiedConfigLoader, get_global_config

class OrderType(Enum):
    """订单类型枚举 - 极致优化"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderStatus(Enum):
    """订单状态枚举 - 极致优化"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"

class ExecutionMode(Enum):
    """执行模式枚举 - 极致优化"""
    REAL_TIME = "real_time"
    SIMULATION = "simulation"
    PAPER_TRADING = "paper_trading"
    BACKTEST = "backtest"

class ExchangeType(Enum):
    """交易所类型枚举 - 极致优化"""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    DERIBIT = "deribit"
    FTX = "ftx"  # 备用
    COINBASE = "coinbase"
    BINGX = "bingx"  # 新增
    BITGET = "bitget"  # 新增
    MEXC = "mexc"  # 新增

@dataclass
class OrderRequest:
    """订单请求数据类 - 极致优化"""
    symbol: str
    order_type: OrderType
    direction: SignalDirection
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    leverage: int = 1
    reduce_only: bool = False
    post_only: bool = False
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_source: str = ""
    signal_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OrderResponse:
    """订单响应数据类 - 极致优化"""
    client_order_id: str
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    execution_latency: float = 0.0  # 执行延迟（毫秒）

@dataclass
class ExecutionMetrics:
    """执行指标数据类 - 极致优化"""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    peak_latency: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)

class UnifiedOrderExecutor(BaseConfigManager):
    """统一订单执行器 V5.0 - 完全重新开发 + 极致优化"""
    
    # 接口元数据 - 极致优化
    _metadata = InterfaceMetadata(
        version="5.0",
        description="统一订单执行器 - 支持FPGA加速、智能路由、成本优化",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "order_execution_time": 0.020,  # 20ms目标
            "order_processing_time": 0.005,
            "routing_decision_time": 0.001
        },
        dependencies=["IRiskManager", "IConfigManager", "IEventDispatcher"],
        compatibility=["4.2", "4.1"]
    )
    
    def __init__(self, config_path: str = None, scope: ConfigScope = ConfigScope.GLOBAL):
        super().__init__(config_path, scope)
        
        # 执行器核心属性
        self.execution_mode = ExecutionMode.REAL_TIME
        self.enabled_exchanges: List[ExchangeType] = []
        self.default_exchange = ExchangeType.BINANCE
        
        # 订单管理
        self._pending_orders: Dict[str, OrderRequest] = {}
        self._order_responses: Dict[str, OrderResponse] = {}
        self._order_history: List[Tuple[OrderRequest, OrderResponse]] = []
        
        # 性能监控
        self._execution_metrics = ExecutionMetrics()
        self._performance_metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0,
            cpu_usage=0.0,
            call_count=0,
            error_count=0,
            cache_hit_rate=0.0
        )
        
        # 智能路由系统
        self._routing_strategies: Dict[str, Callable] = {}
        self._exchange_health: Dict[ExchangeType, Dict[str, Any]] = {}
        self._latency_monitor: Dict[ExchangeType, List[float]] = {}
        
        # 成本优化系统
        self._cost_optimizers: Dict[str, Callable] = {}
        self._slippage_models: Dict[str, Callable] = {}
        
        # 线程安全
        self._order_lock = RLock()
        self._exchange_lock = Lock()
        self._metrics_lock = Lock()
        
        # 异步执行
        self._thread_pool = ThreadPoolExecutor(max_workers=10)
        self._pending_tasks: Dict[str, Future] = {}
        
        # 缓存系统
        self._order_cache: Dict[str, Any] = {}
        self._market_data_cache: Dict[str, Any] = {}
        self._routing_cache: Dict[str, ExchangeType] = {}
        
        # FPGA加速模拟
        self._fpga_enabled = False
        self._fpga_latency_boost = 0.7  # FPGA加速比例
        
        # 智能重试机制
        self._retry_config = {
            "max_retries": 3,
            "retry_delay": 0.1,
            "backoff_factor": 2.0
        }
        
        self.logger = logging.getLogger("engine.order_executor")
        
        # 自动初始化
        self._initialize_execution_engine()
    
    @classmethod
    def get_interface_metadata(cls) -> InterfaceMetadata:
        """获取接口元数据 - 极致优化"""
        return cls._metadata
    
    def load_config(self) -> bool:
        """加载配置 - 极致优化版本"""
        try:
            self.logger.info("开始加载订单执行器配置...")
            
            # 加载基础配置
            if not super().load_config():
                self.logger.error("基础配置加载失败")
                return False
            
            # 加载执行器特定配置
            executor_config = self.config.get("order_executor", {})
            
            # 执行模式配置
            execution_mode = executor_config.get("execution_mode", "real_time")
            self.execution_mode = ExecutionMode(execution_mode)
            
            # 交易所配置
            self.enabled_exchanges = [
                ExchangeType(exchange) for exchange in 
                executor_config.get("enabled_exchanges", ["binance"])
            ]
            self.default_exchange = ExchangeType(
                executor_config.get("default_exchange", "binance")
            )
            
            # FPGA配置
            self._fpga_enabled = executor_config.get("fpga_enabled", False)
            
            # 重试配置
            self._retry_config.update(
                executor_config.get("retry_config", {})
            )
            
            # 初始化路由策略
            self._initialize_routing_strategies()
            
            # 初始化成本优化器
            self._initialize_cost_optimizers()
            
            # 初始化交易所健康监控
            self._initialize_exchange_health_monitor()
            
            self.logger.info(
                f"订单执行器配置加载完成: 模式={self.execution_mode.value}, "
                f"交易所={[e.value for e in self.enabled_exchanges]}, "
                f"FPGA加速={self._fpga_enabled}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"订单执行器配置加载失败: {e}")
            self._performance_metrics.error_count += 1
            return False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值 - 极致优化版本"""
        # 优先从执行器特定配置获取
        executor_config = self.config.get("order_executor", {})
        if key in executor_config:
            return executor_config[key]
        
        # 回退到基础配置
        return super().get_config(key, default)
    
    def set_config(self, key: str, value: Any) -> bool:
        """设置配置值 - 极致优化版本"""
        try:
            # 特殊处理执行器特定配置
            if key.startswith("order_executor."):
                config_key = key[15:]  # 移除"order_executor."前缀
                if "order_executor" not in self.config:
                    self.config["order_executor"] = {}
                self.config["order_executor"][config_key] = value
            else:
                super().set_config(key, value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置配置失败 {key}: {e}")
            return False
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """验证配置完整性 - 极致优化版本"""
        errors = []
        
        try:
            # 验证基础配置
            base_valid, base_errors = super().validate_config()
            if not base_valid:
                errors.extend(base_errors)
            
            # 验证执行器特定配置
            executor_config = self.config.get("order_executor", {})
            
            # 验证执行模式
            execution_mode = executor_config.get("execution_mode")
            if execution_mode not in [mode.value for mode in ExecutionMode]:
                errors.append(f"无效的执行模式: {execution_mode}")
            
            # 验证交易所配置
            enabled_exchanges = executor_config.get("enabled_exchanges", [])
            if not enabled_exchanges:
                errors.append("必须启用至少一个交易所")
            
            for exchange in enabled_exchanges:
                if exchange not in [e.value for e in ExchangeType]:
                    errors.append(f"不支持的交易所: {exchange}")
            
            # 验证重试配置
            retry_config = executor_config.get("retry_config", {})
            max_retries = retry_config.get("max_retries", 3)
            if max_retries < 0 or max_retries > 10:
                errors.append(f"重试次数必须在0-10之间: {max_retries}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"配置验证异常: {e}")
            return False, errors
    
    # 🚀 核心订单执行方法
    
    def execute_order(self, order_request: OrderRequest) -> OrderResponse:
        """执行订单 - 极致优化版本"""
        start_time = time.time()
        
        try:
            self.logger.info(f"开始执行订单: {order_request.client_order_id}")
            
            # 验证订单请求
            validation_result = self._validate_order_request(order_request)
            if not validation_result[0]:
                error_msg = validation_result[1]
                self.logger.error(f"订单验证失败: {error_msg}")
                return self._create_error_response(order_request, error_msg)
            
            # 智能路由决策
            target_exchange = self._select_best_exchange(order_request)
            if not target_exchange:
                error_msg = "无法找到合适的交易所"
                self.logger.error(error_msg)
                return self._create_error_response(order_request, error_msg)
            
            # 成本优化
            optimized_request = self._optimize_order_cost(order_request, target_exchange)
            
            # 风险检查
            risk_check = self._perform_risk_check(optimized_request)
            if not risk_check[0]:
                error_msg = risk_check[1]
                self.logger.warning(f"风险检查失败: {error_msg}")
                return self._create_error_response(optimized_request, error_msg)
            
            # 执行订单
            execution_start = time.time()
            order_response = self._execute_on_exchange(optimized_request, target_exchange)
            execution_time = (time.time() - execution_start) * 1000  # 转换为毫秒
            
            # 记录执行延迟
            order_response.execution_latency = execution_time
            
            # 更新性能指标
            self._update_execution_metrics(order_response, execution_time)
            
            # 记录订单历史
            with self._order_lock:
                self._order_history.append((optimized_request, order_response))
                if len(self._order_history) > 1000:  # 限制历史记录大小
                    self._order_history = self._order_history[-1000:]
            
            total_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"订单执行完成: {order_response.client_order_id}, "
                f"状态: {order_response.status.value}, "
                f"延迟: {execution_time:.2f}ms, "
                f"总耗时: {total_time:.2f}ms"
            )
            
            return order_response
            
        except Exception as e:
            self.logger.error(f"订单执行异常: {e}")
            self._performance_metrics.error_count += 1
            return self._create_error_response(order_request, str(e))
    
    async def execute_order_async(self, order_request: OrderRequest) -> OrderResponse:
        """异步执行订单 - 极致优化版本"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.execute_order, order_request
            )
        except Exception as e:
            self.logger.error(f"异步订单执行失败: {e}")
            return self._create_error_response(order_request, str(e))
    
    def execute_signal(self, signal: IStrategySignal, symbol: str, quantity: float) -> OrderResponse:
        """基于信号执行订单 - 极致优化版本"""
        try:
            # 将信号转换为订单请求
            order_request = self._convert_signal_to_order(signal, symbol, quantity)
            
            # 执行订单
            return self.execute_order(order_request)
            
        except Exception as e:
            self.logger.error(f"信号执行失败: {e}")
            return self._create_error_response(
                OrderRequest(symbol, OrderType.MARKET, SignalDirection.NEUTRAL, quantity),
                str(e)
            )
    
    # 🚀 智能路由系统
    
    def _select_best_exchange(self, order_request: OrderRequest) -> Optional[ExchangeType]:
        """选择最佳交易所 - 智能路由算法"""
        try:
            # 生成路由缓存键
            cache_key = self._generate_routing_cache_key(order_request)
            
            # 检查缓存
            if cache_key in self._routing_cache:
                self._performance_metrics.cache_hit_rate += 1
                return self._routing_cache[cache_key]
            
            # 计算各交易所得分
            exchange_scores = {}
            for exchange in self.enabled_exchanges:
                score = self._calculate_exchange_score(exchange, order_request)
                exchange_scores[exchange] = score
            
            # 选择得分最高的交易所
            best_exchange = max(exchange_scores.items(), key=lambda x: x[1])[0]
            
            # 更新缓存
            self._routing_cache[cache_key] = best_exchange
            
            self.logger.debug(f"路由决策: {order_request.symbol} -> {best_exchange.value}")
            return best_exchange
            
        except Exception as e:
            self.logger.error(f"路由决策失败: {e}")
            return self.default_exchange
    
    def _calculate_exchange_score(self, exchange: ExchangeType, order_request: OrderRequest) -> float:
        """计算交易所得分 - 多因子评估"""
        score = 0.0
        
        try:
            # 1. 延迟因子 (40%)
            latency_score = self._get_latency_score(exchange)
            score += latency_score * 0.4
            
            # 2. 流动性因子 (30%)
            liquidity_score = self._get_liquidity_score(exchange, order_request.symbol)
            score += liquidity_score * 0.3
            
            # 3. 成本因子 (20%)
            cost_score = self._get_cost_score(exchange, order_request)
            score += cost_score * 0.2
            
            # 4. 健康因子 (10%)
            health_score = self._get_health_score(exchange)
            score += health_score * 0.1
            
            # 5. FPGA加速加成
            if self._fpga_enabled and self._is_exchange_fpga_supported(exchange):
                score *= 1.2  # 20%性能加成
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"交易所评分计算失败 {exchange.value}: {e}")
            return 0.5  # 默认得分
    
    def _get_latency_score(self, exchange: ExchangeType) -> float:
        """获取延迟得分"""
        try:
            latencies = self._latency_monitor.get(exchange, [])
            if not latencies:
                return 0.5
            
            avg_latency = sum(latencies) / len(latencies)
            # 延迟越低得分越高（假设50ms为基准）
            return max(0.0, 1.0 - (avg_latency / 50.0))
        except Exception:
            return 0.5
    
    def _get_liquidity_score(self, exchange: ExchangeType, symbol: str) -> float:
        """获取流动性得分"""
        # 简化实现 - 实际中需要实时流动性数据
        liquidity_tiers = {
            "BTCUSDT": 0.9,
            "ETHUSDT": 0.8,
            "BNBUSDT": 0.7
        }
        return liquidity_tiers.get(symbol, 0.5)
    
    def _get_cost_score(self, exchange: ExchangeType, order_request: OrderRequest) -> float:
        """获取成本得分"""
        # 简化实现 - 实际中需要实时费率数据
        fee_structures = {
            ExchangeType.BINANCE: 0.1,  # 0.1% 费率
            ExchangeType.BYBIT: 0.075,  # 0.075% 费率
            ExchangeType.OKX: 0.08,     # 0.08% 费率
            ExchangeType.BINGX: 0.085,  # 新增: 0.085% 费率
            ExchangeType.BITGET: 0.082, # 新增: 0.082% 费率
            ExchangeType.MEXC: 0.088,   # 新增: 0.088% 费率
        }
        base_fee = fee_structures.get(exchange, 0.1)
        
        # 根据订单类型调整
        if order_request.order_type == OrderType.LIMIT:
            base_fee *= 0.5  # 限价单通常有费率优惠
        
        # 成本越低得分越高
        return max(0.0, 1.0 - (base_fee / 0.1))
    
    def _get_health_score(self, exchange: ExchangeType) -> float:
        """获取健康得分"""
        health_data = self._exchange_health.get(exchange, {})
        return health_data.get("health_score", 0.5)
    
    # 🚀 成本优化系统
    
    def _optimize_order_cost(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderRequest:
        """优化订单成本 - 极致优化"""
        try:
            optimized_request = order_request
            
            # 应用滑点优化
            optimized_request = self._apply_slippage_optimization(optimized_request, exchange)
            
            # 应用手续费优化
            optimized_request = self._apply_fee_optimization(optimized_request, exchange)
            
            # 应用执行优化
            optimized_request = self._apply_execution_optimization(optimized_request, exchange)
            
            return optimized_request
            
        except Exception as e:
            self.logger.warning(f"成本优化失败: {e}")
            return order_request  # 返回原始请求作为降级方案
    
    def _apply_slippage_optimization(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderRequest:
        """应用滑点优化"""
        # 根据市场深度和波动率调整价格
        if order_request.order_type == OrderType.MARKET:
            # 对于市价单，使用更激进的滑点保护
            slippage_factor = self._calculate_slippage_factor(order_request, exchange)
            if order_request.direction == SignalDirection.LONG:
                # 买入时适当提高价格
                if order_request.price:
                    order_request.price *= (1 + slippage_factor)
            else:
                # 卖出时适当降低价格
                if order_request.price:
                    order_request.price *= (1 - slippage_factor)
        
        return order_request
    
    def _apply_fee_optimization(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderRequest:
        """应用手续费优化"""
        # 根据交易所费率结构优化订单参数
        if order_request.order_type == OrderType.LIMIT:
            # 限价单可以设置post_only避免成为taker
            order_request.post_only = True
        
        return order_request
    
    def _apply_execution_optimization(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderRequest:
        """应用执行优化"""
        # 大额订单使用冰山订单或TWAP执行
        large_order_threshold = self.get_config("large_order_threshold", 10000.0)
        
        if order_request.quantity * (order_request.price or 1.0) > large_order_threshold:
            if order_request.order_type == OrderType.MARKET:
                # 大额市价单转换为TWAP执行
                order_request.order_type = OrderType.TWAP
                order_request.metadata["execution_algorithm"] = "twap"
            else:
                # 大额限价单使用冰山订单
                order_request.order_type = OrderType.ICEBERG
                order_request.metadata["iceberg_parts"] = 5
        
        return order_request
    
    # 🚀 风险控制系统
    
    def _perform_risk_check(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """执行风险检查 - 预测性风控"""
        try:
            # 1. 数量验证
            if order_request.quantity <= 0:
                return False, "订单数量必须大于0"
            
            # 2. 价格验证
            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if not order_request.price or order_request.price <= 0:
                    return False, "限价单必须指定有效价格"
            
            # 3. 杠杆验证
            if order_request.leverage < 1 or order_request.leverage > 100:
                return False, "杠杆必须在1-100之间"
            
            # 4. 市场状态检查
            market_state = self._get_market_state(order_request.symbol)
            if market_state.get("volatility", 0) > 0.1:  # 高波动率
                if order_request.order_type == OrderType.MARKET:
                    return False, "高波动率市场禁止使用市价单"
            
            # 5. 集中度风险检查
            concentration = self._calculate_position_concentration(order_request)
            if concentration > 0.8:  # 80%集中度限制
                return False, "仓位集中度过高"
            
            return True, "风险检查通过"
            
        except Exception as e:
            self.logger.error(f"风险检查异常: {e}")
            return False, f"风险检查异常: {e}"
    
    def _get_market_state(self, symbol: str) -> Dict[str, Any]:
        """获取市场状态"""
        # 简化实现 - 实际中需要实时市场数据
        return {
            "volatility": 0.05,
            "liquidity": "high",
            "regime": MarketRegime.NORMAL
        }
    
    def _calculate_position_concentration(self, order_request: OrderRequest) -> float:
        """计算仓位集中度"""
        # 简化实现 - 实际中需要持仓数据
        return 0.3
    
    # 🚀 交易所执行引擎
    
    def _execute_on_exchange(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderResponse:
        """在指定交易所执行订单"""
        start_time = time.time()
        
        try:
            # 模拟FPGA加速
            if self._fpga_enabled:
                execution_delay = self._simulate_fpga_execution()
            else:
                execution_delay = self._simulate_normal_execution()
            
            # 模拟网络延迟
            network_delay = self._simulate_network_delay(exchange)
            
            # 总延迟
            total_delay = execution_delay + network_delay
            
            # 模拟执行结果
            if total_delay < 0.1:  # 100ms内大概率成功
                success_probability = 0.95
            else:
                success_probability = 0.8
            
            import random
            if random.random() < success_probability:
                return self._create_success_response(order_request, exchange)
            else:
                return self._create_error_response(order_request, "交易所执行失败")
                
        except Exception as e:
            self.logger.error(f"交易所执行异常 {exchange.value}: {e}")
            return self._create_error_response(order_request, str(e))
    
    def _simulate_fpga_execution(self) -> float:
        """模拟FPGA加速执行"""
        # FPGA加速减少70%延迟
        base_delay = 0.005  # 5ms基础延迟
        return base_delay * self._fpga_latency_boost
    
    def _simulate_normal_execution(self) -> float:
        """模拟正常执行"""
        base_delay = 0.005  # 5ms基础延迟
        variation = 0.002   # 2ms波动
        import random
        return base_delay + random.uniform(0, variation)
    
    def _simulate_network_delay(self, exchange: ExchangeType) -> float:
        """模拟网络延迟"""
        # 基于交易所地理位置和网络状况的延迟
        exchange_delays = {
            ExchangeType.BINANCE: 0.015,  # 15ms
            ExchangeType.BYBIT: 0.020,    # 20ms
            ExchangeType.OKX: 0.018,      # 18ms
            ExchangeType.DERIBIT: 0.025,  # 25ms
            ExchangeType.BINGX: 0.022,    # 新增: 22ms
            ExchangeType.BITGET: 0.024,   # 新增: 24ms
            ExchangeType.MEXC: 0.026,     # 新增: 26ms
        }
        base_delay = exchange_delays.get(exchange, 0.020)
        
        # 添加随机波动
        import random
        variation = base_delay * 0.3  # 30%波动
        return base_delay + random.uniform(-variation, variation)
    
    # 🚀 响应创建方法
    
    def _create_success_response(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderResponse:
        """创建成功响应"""
        return OrderResponse(
            client_order_id=order_request.client_order_id,
            exchange_order_id=f"{exchange.value}_{uuid.uuid4().hex[:8]}",
            status=OrderStatus.FILLED,
            filled_quantity=order_request.quantity,
            average_price=order_request.price or self._get_current_price(order_request.symbol),
            commission=order_request.quantity * 0.001,  # 0.1%手续费
            commission_asset="USDT",
            timestamp=datetime.now()
        )
    
    def _create_error_response(self, order_request: OrderRequest, error_message: str) -> OrderResponse:
        """创建错误响应"""
        return OrderResponse(
            client_order_id=order_request.client_order_id,
            status=OrderStatus.ERROR,
            error_message=error_message,
            timestamp=datetime.now()
        )
    
    # 🚀 工具方法
    
    def _validate_order_request(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """验证订单请求"""
        if not order_request.symbol:
            return False, "交易对不能为空"
        
        if order_request.quantity <= 0:
            return False, "订单数量必须大于0"
        
        if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if not order_request.price or order_request.price <= 0:
                return False, "限价单必须指定有效价格"
        
        if order_request.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
            if not order_request.stop_price or order_request.stop_price <= 0:
                return False, "止损单必须指定有效止损价格"
        
        return True, "验证通过"
    
    def _convert_signal_to_order(self, signal: IStrategySignal, symbol: str, quantity: float) -> OrderRequest:
        """将信号转换为订单请求"""
        order_type = OrderType.MARKET  # 默认市价单
        
        # 根据信号置信度选择订单类型
        confidence = signal.get_confidence()
        if confidence > 0.8:
            order_type = OrderType.LIMIT  # 高置信度使用限价单
        
        return OrderRequest(
            symbol=symbol,
            order_type=order_type,
            direction=signal.get_signal_direction(),
            quantity=quantity,
            price=self._get_current_price(symbol) if order_type == OrderType.LIMIT else None,
            strategy_source=type(signal).__name__,
            signal_confidence=confidence,
            metadata={
                "signal_timestamp": signal.get_timestamp(),
                "signal_strength": signal.get_signal_strength()
            }
        )
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前价格 - 简化实现"""
        # 实际中需要从市场数据流获取
        price_map = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "BNBUSDT": 500.0
        }
        return price_map.get(symbol, 100.0)
    
    def _generate_routing_cache_key(self, order_request: OrderRequest) -> str:
        """生成路由缓存键"""
        key_data = {
            "symbol": order_request.symbol,
            "order_type": order_request.order_type.value,
            "direction": order_request.direction.value,
            "quantity": order_request.quantity
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _is_exchange_fpga_supported(self, exchange: ExchangeType) -> bool:
        """检查交易所是否支持FPGA加速"""
        fpga_supported = [
            ExchangeType.BINANCE, 
            ExchangeType.BYBIT, 
            ExchangeType.OKX,
            ExchangeType.BINGX,  # 新增支持
            ExchangeType.BITGET  # 新增支持
        ]
        return exchange in fpga_supported
    
    def _calculate_slippage_factor(self, order_request: OrderRequest, exchange: ExchangeType) -> float:
        """计算滑点因子"""
        # 基于订单大小和市场深度的滑点估计
        base_slippage = 0.001  # 0.1%基础滑点
        
        # 大额订单增加滑点
        size_factor = min(1.0, order_request.quantity / 1000.0)  # 假设1000为基准
        base_slippage *= (1 + size_factor * 0.5)  # 大额订单增加50%滑点
        
        return base_slippage
    
    def _update_execution_metrics(self, order_response: OrderResponse, execution_time: float):
        """更新执行指标"""
        with self._metrics_lock:
            self._execution_metrics.total_orders += 1
            
            if order_response.status == OrderStatus.FILLED:
                self._execution_metrics.successful_orders += 1
            else:
                self._execution_metrics.failed_orders += 1
            
            # 更新平均执行时间
            current_avg = self._execution_metrics.average_execution_time
            total_orders = self._execution_metrics.total_orders
            self._execution_metrics.average_execution_time = (
                (current_avg * (total_orders - 1) + execution_time) / total_orders
            )
            
            # 更新成功率
            self._execution_metrics.success_rate = (
                self._execution_metrics.successful_orders / self._execution_metrics.total_orders
            )
            
            # 更新峰值延迟
            if execution_time > self._execution_metrics.peak_latency:
                self._execution_metrics.peak_latency = execution_time
    
    # 🚀 系统初始化方法
    
    def _initialize_execution_engine(self):
        """初始化执行引擎"""
        try:
            self.logger.info("初始化订单执行引擎...")
            
            # 初始化路由策略
            self._initialize_routing_strategies()
            
            # 初始化成本优化器
            self._initialize_cost_optimizers()
            
            # 初始化交易所健康监控
            self._initialize_exchange_health_monitor()
            
            # 加载配置
            self.load_config()
            
            self.logger.info("订单执行引擎初始化完成")
            
        except Exception as e:
            self.logger.error(f"执行引擎初始化失败: {e}")
    
    def _initialize_routing_strategies(self):
        """初始化路由策略"""
        self._routing_strategies = {
            "latency_priority": self._latency_priority_routing,
            "cost_priority": self._cost_priority_routing,
            "balanced": self._balanced_routing
        }
    
    def _initialize_cost_optimizers(self):
        """初始化成本优化器"""
        self._cost_optimizers = {
            "slippage_reduction": self._optimize_slippage,
            "fee_minimization": self._optimize_fees,
            "execution_improvement": self._improve_execution
        }
    
    def _initialize_exchange_health_monitor(self):
        """初始化交易所健康监控"""
        for exchange in ExchangeType:
            self._exchange_health[exchange] = {
                "health_score": 1.0,
                "last_check": datetime.now(),
                "error_count": 0,
                "success_count": 0
            }
            self._latency_monitor[exchange] = []
    
    # 🚀 路由策略实现
    
    def _latency_priority_routing(self, order_request: OrderRequest) -> ExchangeType:
        """延迟优先路由"""
        lowest_latency = float('inf')
        best_exchange = self.default_exchange
        
        for exchange in self.enabled_exchanges:
            latency = self._get_average_latency(exchange)
            if latency < lowest_latency:
                lowest_latency = latency
                best_exchange = exchange
        
        return best_exchange
    
    def _cost_priority_routing(self, order_request: OrderRequest) -> ExchangeType:
        """成本优先路由"""
        lowest_cost = float('inf')
        best_exchange = self.default_exchange
        
        for exchange in self.enabled_exchanges:
            cost = self._estimate_transaction_cost(order_request, exchange)
            if cost < lowest_cost:
                lowest_cost = cost
                best_exchange = exchange
        
        return best_exchange
    
    def _balanced_routing(self, order_request: OrderRequest) -> ExchangeType:
        """平衡路由策略"""
        best_score = -1
        best_exchange = self.default_exchange
        
        for exchange in self.enabled_exchanges:
            latency_score = self._get_latency_score(exchange)
            cost_score = self._get_cost_score(exchange, order_request)
            health_score = self._get_health_score(exchange)
            
            # 加权得分
            total_score = (latency_score * 0.4 + cost_score * 0.4 + health_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_exchange = exchange
        
        return best_exchange
    
    def _get_average_latency(self, exchange: ExchangeType) -> float:
        """获取平均延迟"""
        latencies = self._latency_monitor.get(exchange, [])
        return sum(latencies) / len(latencies) if latencies else 100.0  # 默认100ms
    
    def _estimate_transaction_cost(self, order_request: OrderRequest, exchange: ExchangeType) -> float:
        """估算交易成本"""
        # 简化实现
        fee_rates = {
            ExchangeType.BINANCE: 0.001,
            ExchangeType.BYBIT: 0.00075,
            ExchangeType.OKX: 0.0008,
            ExchangeType.BINGX: 0.00085,  # 新增
            ExchangeType.BITGET: 0.00082, # 新增
            ExchangeType.MEXC: 0.00088,   # 新增
        }
        base_fee = fee_rates.get(exchange, 0.001)
        
        order_value = order_request.quantity * (order_request.price or 1.0)
        return order_value * base_fee
    
    # 🚀 成本优化器实现
    
    def _optimize_slippage(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderRequest:
        """优化滑点"""
        # 实现滑点优化逻辑
        return order_request
    
    def _optimize_fees(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderRequest:
        """优化手续费"""
        # 实现手续费优化逻辑
        return order_request
    
    def _improve_execution(self, order_request: OrderRequest, exchange: ExchangeType) -> OrderRequest:
        """改进执行"""
        # 改进执行逻辑
        return order_request
    
    # 🚀 性能监控和报告
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        # 更新缓存命中率
        total_operations = self._performance_metrics.call_count
        if total_operations > 0:
            self._performance_metrics.cache_hit_rate = self._performance_metrics.cache_hit_rate / total_operations
        
        return self._performance_metrics
    
    def get_execution_metrics(self) -> ExecutionMetrics:
        """获取执行指标"""
        return self._execution_metrics
    
    def get_order_history(self, limit: int = 100) -> List[Tuple[OrderRequest, OrderResponse]]:
        """获取订单历史"""
        with self._order_lock:
            return self._order_history[-limit:]
    
    def reset_metrics(self):
        """重置指标"""
        with self._metrics_lock:
            self._execution_metrics = ExecutionMetrics()
            self._performance_metrics = PerformanceMetrics(
                execution_time=0.0,
                memory_usage=0,
                cpu_usage=0.0,
                call_count=0,
                error_count=0,
                cache_hit_rate=0.0
            )
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """获取详细状态"""
        return {
            "execution_mode": self.execution_mode.value,
            "enabled_exchanges": [e.value for e in self.enabled_exchanges],
            "fpga_enabled": self._fpga_enabled,
            "total_orders": self._execution_metrics.total_orders,
            "success_rate": self._execution_metrics.success_rate,
            "average_latency": self._execution_metrics.average_execution_time,
            "performance_metrics": self.get_performance_metrics().to_dict(),
            "exchange_health": {
                exchange.value: health_data 
                for exchange, health_data in self._exchange_health.items()
            }
        }

# 全局订单执行器实例
_global_order_executor: Optional[UnifiedOrderExecutor] = None

def get_global_order_executor() -> UnifiedOrderExecutor:
    """获取全局订单执行器"""
    global _global_order_executor
    
    if _global_order_executor is None:
        _global_order_executor = UnifiedOrderExecutor()
        _global_order_executor.load_config()
    
    return _global_order_executor

# 自动注册接口
from src.interfaces import InterfaceRegistry
InterfaceRegistry.register_interface(UnifiedOrderExecutor)

__all__ = [
    'UnifiedOrderExecutor',
    'OrderType',
    'OrderStatus', 
    'ExecutionMode',
    'ExchangeType',
    'OrderRequest',
    'OrderResponse',
    'ExecutionMetrics',
    'get_global_order_executor'
]



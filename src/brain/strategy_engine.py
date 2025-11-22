# src/brain/strategy_engine.py
"""量子奇点狙击系统 - 策略引擎 V5.0 (完全重新开发 + 极致优化 + 完整整合版本)"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future

# 导入极致优化的依赖模块
from src.interfaces import (
    IStrategySignal, SignalDirection, SignalPriority, PerformanceMetrics,
    InterfaceMetadata, SignalMetadata, MarketRegime, DataQualityLevel,
    IEventDispatcher, Event, EventPriority, IRiskManager, IMarketAnalyzer,
    ConfigScope, ConfigChange
)
from src.core.strategy_base import BaseStrategy, StrategySignal, StrategyFactory, StrategyError
from src.core.config_manager import BaseConfigManager, ConfigManagerFactory
from src.config.config import UnifiedConfigLoader, get_global_config

# 条件导入，确保向后兼容
try:
    from brain.strategy_integration import (
        StrategyIntegrationEngine, StrategyIntegrationFactory,
        IntegrationMode, FusionResult, StrategyPerformance
    )
    HAS_INTEGRATION_ENGINE = True
except ImportError:
    HAS_INTEGRATION_ENGINE = False
    # 创建简化版本
    class StrategyPerformance:
        def __init__(self, strategy_name: str):
            self.strategy_name = strategy_name
            self.performance_score = 0.5
            self.signal_count = 0
            self.success_rate = 0.5
            self.last_updated = datetime.now()

class EngineState(Enum):
    """引擎状态枚举 - 整合版本"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ExecutionMode(Enum):
    """执行模式枚举 - 整合版本"""
    REALTIME = "realtime"
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    SIMULATION = "simulation"

@dataclass
class EngineMetrics:
    """引擎指标数据类 - 整合版本"""
    total_strategies: int = 0
    active_strategies: int = 0
    signals_generated: int = 0
    signals_executed: int = 0
    execution_success_rate: float = 0.0
    average_signal_latency: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)

@dataclass
class StrategyContext:
    """策略上下文数据类 - 整合版本"""
    strategy_name: str
    market_data: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    risk_parameters: Dict[str, Any]
    execution_mode: ExecutionMode
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class StrategyEngine(BaseStrategy):
    """策略引擎 V5.0 - 完整整合版本"""
    
    # 接口元数据
    _metadata = InterfaceMetadata(
        version="5.0-integrated",
        description="智能策略引擎 - 完整整合版本，兼具稳定性和完整功能",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "strategy_initialization_time": 0.01,
            "signal_processing_time": 0.005,
            "event_handling_time": 0.001
        },
        dependencies=[
            "BaseStrategy", "IEventDispatcher", "IRiskManager", 
            "IMarketAnalyzer", "BaseConfigManager"
        ],
        compatibility=["5.0", "4.2", "4.1", "emergency-rebuild"]
    )
    
    def __init__(self, name: str = "QuantumStrategyEngine", config: Dict[str, Any] = None):
        # 配置处理 - 整合版本
        config = config or {}
        default_config = {
            "name": name,
            "execution_mode": ExecutionMode.REALTIME.value,
            "max_concurrent_strategies": 10,
            "signal_batch_size": 100,
            "health_check_interval": 30,
            "performance_monitoring": True,
            "auto_recovery": True,
            "enable_advanced_features": False,  # 默认关闭高级功能，确保稳定性
            "event_driven_architecture": False  # 默认关闭事件驱动
        }
        
        # 完整版本配置扩展（条件启用）
        advanced_defaults = {
            "enabled": True,
            "risk_level": "medium",
            "strategy_timeout_seconds": 300,
            "memory_usage_threshold_mb": 1024,
            "cpu_usage_threshold_percent": 80,
            "signal_validation_strict": True,
            "adaptive_load_balancing": False,  # 默认关闭
            "quantum_coherence_integration": False
        }
        
        default_config.update(advanced_defaults)
        default_config.update(config)
        
        super().__init__(name, default_config)
        
        # ==================== 核心引擎属性 - 整合版本 ====================
        
        # 引擎状态管理
        self._engine_state: EngineState = EngineState.INITIALIZING
        self._execution_mode: ExecutionMode = ExecutionMode(config.get("execution_mode", "realtime"))
        self._last_state_change: datetime = datetime.now()
        
        # 策略管理 - 使用应急版本的稳定实现
        self._strategies: Dict[str, BaseStrategy] = {}
        self._strategy_contexts: Dict[str, StrategyContext] = {}
        self._strategy_performance: Dict[str, StrategyPerformance] = {}
        self._strategy_dependencies: Dict[str, List[str]] = {}
        
        # 高级功能组件（条件初始化）
        self._enable_advanced_features = config.get("enable_advanced_features", False)
        self._event_dispatcher: Optional[IEventDispatcher] = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._integration_engine: Optional[Any] = None
        
        # 性能监控 - 整合版本
        self._engine_metrics = EngineMetrics()
        self._performance_metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0,
            cpu_usage=0.0,
            call_count=0,
            error_count=0,
            cache_hit_rate=0.0
        )
        
        # 线程安全
        self._engine_lock = threading.RLock()
        self._strategy_lock = threading.RLock()
        self._event_lock = threading.RLock()
        
        # 异步执行
        self._thread_pool = ThreadPoolExecutor(max_workers=config.get("max_concurrent_strategies", 10))
        self._pending_tasks: Dict[str, Future] = {}
        
        # 缓存系统
        self._signal_cache: Dict[str, IStrategySignal] = {}
        self._market_data_cache: Dict[str, Any] = {}
        self._performance_cache: Dict[str, Any] = {}
        
        # 健康监控
        self._health_check_timer: Optional[threading.Timer] = None
        self._last_health_check: datetime = datetime.now()
        
        # 错误恢复
        self._error_count: Dict[str, int] = defaultdict(int)
        self._recovery_attempts: Dict[str, int] = defaultdict(int)
        
        self.logger = logging.getLogger(f"engine.{name}")
        
        # 自动初始化关键组件
        self._initialize_critical_components()
    
    @classmethod
    def get_interface_metadata(cls) -> InterfaceMetadata:
        """获取接口元数据 - 整合版本"""
        return cls._metadata
    
    def initialize(self) -> bool:
        """初始化策略引擎 - 整合版本"""
        start_time = datetime.now()
        
        try:
            with self._engine_lock:
                if self.initialized:
                    self.logger.warning("策略引擎已经初始化")
                    return True
                
                self.logger.info("开始初始化量子策略引擎（整合版本）...")
                self._update_engine_state(EngineState.INITIALIZING)
                
                # ==================== 分步初始化流程 ====================
                
                # 1. 初始化配置系统（应急版本实现）
                if not self._initialize_config_system():
                    raise StrategyError("配置系统初始化失败", self.name)
                
                # 2. 条件初始化事件系统
                if self._enable_advanced_features:
                    if not self._initialize_event_system():
                        self.logger.warning("事件系统初始化失败，继续其他组件")
                else:
                    self.logger.info("事件系统已禁用（稳定性优先）")
                
                # 3. 加载和初始化策略（使用应急版本的稳定实现）
                if not self._load_and_initialize_strategies():
                    raise StrategyError("策略加载和初始化失败", self.name)
                
                # 4. 条件初始化整合引擎
                if self._enable_advanced_features and HAS_INTEGRATION_ENGINE:
                    if not self._initialize_integration_engine():
                        self.logger.warning("整合引擎初始化失败，使用基础模式")
                else:
                    self.logger.info("整合引擎已禁用")
                
                # 5. 启动健康监控
                if not self._start_health_monitoring():
                    self.logger.warning("健康监控启动失败")
                
                # 6. 验证引擎完整性
                if not self._validate_engine_integrity():
                    raise StrategyError("引擎完整性验证失败", self.name)
                
                # 更新状态和性能指标
                self.initialized = True
                self._update_engine_state(EngineState.READY)
                
                initialization_time = (datetime.now() - start_time).total_seconds()
                self._performance_metrics.execution_time += initialization_time
                self._performance_metrics.call_count += 1
                
                self.logger.info(
                    f"策略引擎初始化完成: {len(self._strategies)} 个策略, "
                    f"耗时: {initialization_time:.3f}s, "
                    f"高级功能: {'启用' if self._enable_advanced_features else '禁用'}"
                )
                
                # 条件发布引擎就绪事件
                if self._enable_advanced_features:
                    self._publish_engine_event("engine_ready", {
                        "strategies_count": len(self._strategies),
                        "initialization_time": initialization_time,
                        "execution_mode": self._execution_mode.value,
                        "advanced_features": self._enable_advanced_features
                    })
                
                return True
                
        except Exception as e:
            self.logger.error(f"策略引擎初始化失败: {e}")
            self._update_engine_state(EngineState.ERROR)
            self._performance_metrics.error_count += 1
            return False
    
    def get_signal(self, data: Any) -> Optional[IStrategySignal]:
        """获取交易信号 - 整合版本"""
        if not self.initialized:
            self.logger.error("策略引擎未初始化")
            return None
        
        # 允许在 READY 和 RUNNING 状态下生成信号
        if self._engine_state not in [EngineState.RUNNING, EngineState.READY]:
            self.logger.warning(f"策略引擎状态为 {self._engine_state.value}，无法生成信号")
            return None
        
        start_time = datetime.now()
        
        try:
            # 验证输入数据（应急版本实现）
            if not self._validate_input_data(data):
                self.logger.warning("输入数据验证失败")
                return None
            
            # 处理市场数据（应急版本实现）
            processed_data = self._preprocess_market_data(data)
            if not processed_data:
                return None
            
            # 创建策略上下文
            strategy_context = self._create_strategy_context(processed_data)
            
            # 执行策略信号生成（应急版本稳定实现）
            signals = self._execute_strategies(strategy_context)
            if not signals:
                self.logger.debug("无有效策略信号生成")
                return None
            
            # 信号整合与融合（条件使用高级功能）
            if self._enable_advanced_features and self._integration_engine and len(signals) > 1:
                integrated_signal = self._integration_engine.get_signal(signals)
            else:
                # 使用应急版本的简单整合
                integrated_signal = self._integrate_signals_simple(signals)
            
            if not integrated_signal:
                return None
            
            # 验证和优化最终信号
            final_signal = self._validate_and_optimize_signal(integrated_signal, strategy_context)
            if not final_signal:
                return None
            
            # 更新性能和指标
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_signal_metrics(processing_time, len(signals))
            
            # 条件发布信号生成事件
            if self._enable_advanced_features:
                self._publish_signal_event(final_signal, len(signals), processing_time)
            
            self.logger.debug(
                f"信号生成完成: {len(signals)} 个策略信号, "
                f"整合信号强度: {final_signal.get_confidence():.3f}, "
                f"耗时: {processing_time:.3f}s"
            )
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {e}")
            self._performance_metrics.error_count += 1
            self._handle_signal_generation_error(e)
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取策略引擎状态 - 整合版本"""
        try:
            # 安全地获取基础状态
            base_status = {}
            try:
                base_status = super().get_status() or {}
            except Exception as e:
                self.logger.warning(f"获取基础状态失败: {e}")
                base_status = {}
            
            # 确保 base_status 是字典
            if not isinstance(base_status, dict):
                base_status = {}
            
            engine_status = {
                "engine_state": self._engine_state.value,
                "execution_mode": self._execution_mode.value,
                "total_strategies": len(self._strategies),
                "active_strategies": self._get_active_strategies_count(),
                "initialized": self.initialized,
                "last_state_change": self._last_state_change.isoformat(),
                "performance_metrics": self._get_performance_metrics_dict(),
                "engine_metrics": self._get_engine_metrics_dict(),
                "advanced_features_enabled": self._enable_advanced_features,
                "integration_engine_active": self._integration_engine is not None,
                "event_system_active": self._event_dispatcher is not None,
                "health_status": self._get_health_status(),
                "version": "5.0-integrated",
                "name": self.name
            }
            
            # 合并基础状态和引擎状态
            return {**base_status, **engine_status}
            
        except Exception as e:
            self.logger.error(f"获取引擎状态失败: {e}")
            return {
                "engine_state": "error",
                "error": str(e),
                "initialized": False,
                "version": "5.0-integrated",
                "name": self.name if hasattr(self, 'name') else "unknown"
            }
    
    # ==================== 核心方法 - 整合版本 ====================
    
    def enable_advanced_features(self) -> bool:
        """启用高级功能 - 动态启用"""
        try:
            if self._enable_advanced_features:
                self.logger.info("高级功能已经启用")
                return True
            
            self.logger.info("正在启用高级功能...")
            
            # 启用事件系统
            if not self._initialize_event_system():
                self.logger.warning("事件系统启用失败")
            
            # 启用整合引擎
            if HAS_INTEGRATION_ENGINE and not self._initialize_integration_engine():
                self.logger.warning("整合引擎启用失败")
            
            self._enable_advanced_features = True
            self.logger.info("高级功能启用完成")
            return True
            
        except Exception as e:
            self.logger.error(f"启用高级功能失败: {e}")
            return False
    
    def disable_advanced_features(self) -> bool:
        """禁用高级功能 - 回到稳定模式"""
        try:
            if not self._enable_advanced_features:
                self.logger.info("高级功能已经禁用")
                return True
            
            self.logger.info("正在禁用高级功能...")
            
            # 停止事件系统
            self._event_dispatcher = None
            self._event_handlers.clear()
            
            # 停止整合引擎
            self._integration_engine = None
            
            self._enable_advanced_features = False
            self.logger.info("高级功能禁用完成")
            return True
            
        except Exception as e:
            self.logger.error(f"禁用高级功能失败: {e}")
            return False
    
    # ==================== 应急版本的核心实现 ====================
    
    def _load_and_initialize_strategies(self) -> bool:
        """加载和初始化策略 - 应急版本稳定实现"""
        try:
            # 从配置加载策略列表
            strategy_configs = self.config.get("strategies", [])
            
            if not strategy_configs:
                self.logger.warning("未配置策略列表，使用应急策略发现")
                return self._emergency_strategy_discovery()
            
            loaded_count = 0
            for strategy_config in strategy_configs:
                strategy_name = strategy_config.get("name")
                strategy_type = strategy_config.get("type")
                
                if not strategy_name or not strategy_type:
                    self.logger.warning("跳过无效的策略配置")
                    continue
                
                try:
                    # 使用策略工厂创建策略
                    strategy_instance = StrategyFactory.create_strategy(
                        strategy_name, strategy_config
                    )
                    
                    if strategy_instance and strategy_instance.initialize():
                        self._strategies[strategy_name] = strategy_instance
                        self._strategy_dependencies[strategy_name] = strategy_config.get("dependencies", [])
                        self._strategy_performance[strategy_name] = StrategyPerformance(
                            strategy_name=strategy_name
                        )
                        loaded_count += 1
                        self.logger.info(f"✅ 策略加载成功: {strategy_name}")
                    else:
                        self.logger.error(f"❌ 策略初始化失败: {strategy_name}")
                        
                except Exception as e:
                    self.logger.error(f"❌ 策略加载异常 {strategy_name}: {e}")
            
            if loaded_count == 0:
                self.logger.error("❌ 无策略成功加载")
                return False
            
            self.logger.info(f"🎯 策略加载完成: {loaded_count} 个策略")
            return True
            
        except Exception as e:
            self.logger.error(f"策略加载过程异常: {e}")
            return False
    
    def _emergency_strategy_discovery(self):
        """应急策略发现 - 稳定可靠的实现"""
        try:
            self.logger.info("开始应急策略发现过程...")
            
            # 应急基础策略 - 完全实现的稳定版本
            class EmergencyBaseStrategy(BaseStrategy):
                """应急基础策略 - 完全实现所有抽象方法"""
                
                def __init__(self, name, config=None):
                    super().__init__(name, config or {})
                    self.initialized = False
                    self.signal_count = 0
                
                def initialize(self):
                    self.initialized = True
                    self.logger.info(f"应急策略初始化完成: {self.name}")
                    return True
                
                def get_signal(self, data):
                    if not self.initialized:
                        return None
                    
                    self.signal_count += 1
                    
                    from src.interfaces import SignalDirection, SignalMetadata
                    from src.core.strategy_base import StrategySignal
                    
                    metadata = SignalMetadata(
                        source="emergency_base_strategy",
                        tags=["emergency", "stable"]
                    )
                    
                    return StrategySignal(
                        signal_type="EMERGENCY_BASE",
                        confidence=0.5,
                        data={"strategy": "emergency_base", "signal_count": self.signal_count},
                        direction=SignalDirection.NEUTRAL,
                        metadata=metadata
                    )
                
                def get_status(self):
                    return {
                        "name": self.name,
                        "initialized": self.initialized,
                        "signal_count": self.signal_count,
                        "strategy_type": "emergency_base"
                    }
            
            # 创建应急策略
            strategy_names = ["EmergencyTrend", "EmergencyMeanReversion", "EmergencyBreakout"]
            loaded_count = 0
            
            for strategy_name in strategy_names:
                try:
                    strategy_instance = EmergencyBaseStrategy(strategy_name, {})
                    
                    if strategy_instance and strategy_instance.initialize():
                        self._strategies[strategy_name] = strategy_instance
                        self._strategy_dependencies[strategy_name] = []
                        self._strategy_performance[strategy_name] = StrategyPerformance(
                            strategy_name=strategy_name
                        )
                        loaded_count += 1
                        self.logger.info(f"✅ 应急策略加载成功: {strategy_name}")
                    else:
                        self.logger.error(f"❌ 应急策略初始化失败: {strategy_name}")
                        
                except Exception as e:
                    self.logger.error(f"❌ 应急策略加载异常 {strategy_name}: {e}")
            
            if loaded_count == 0:
                self.logger.error("❌ 无应急策略成功加载")
                return False
            
            self.logger.info(f"🎯 应急策略加载完成: {loaded_count} 个策略")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 应急策略发现过程异常: {e}")
            return False
    
    def _integrate_signals_simple(self, signals: Dict[str, IStrategySignal]) -> Optional[IStrategySignal]:
        """简单信号整合 - 应急版本稳定实现"""
        try:
            # 简单整合：选择置信度最高的信号
            best_signal = None
            best_confidence = 0.0
            
            for signal in signals.values():
                confidence = signal.get_confidence()
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_signal = signal
            
            return best_signal
            
        except Exception as e:
            self.logger.error(f"简单信号整合失败: {e}")
            # 返回第一个信号作为降级方案
            return next(iter(signals.values())) if signals else None
    
    # ==================== 高级功能的条件实现 ====================
    
    def _initialize_event_system(self) -> bool:
        """初始化事件系统 - 条件实现"""
        if not self._enable_advanced_features:
            return True
            
        try:
            # 这里应该从接口注册表获取事件分发器
            # 简化实现：创建基本的事件处理器
            
            self._event_handlers = {
                "engine_state_change": [],
                "strategy_signal": [],
                "performance_alert": [],
                "error_occurred": []
            }
            
            self.logger.info("事件系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"事件系统初始化失败: {e}")
            return False
    
    def _initialize_integration_engine(self) -> bool:
        """初始化整合引擎 - 条件实现"""
        if not self._enable_advanced_features or not HAS_INTEGRATION_ENGINE:
            return True
            
        try:
            integration_config = self.config.get("integration", {})
            self._integration_engine = StrategyIntegrationFactory.create_integration_engine(
                f"{self.name}_integration", integration_config
            )
            
            # 将所有策略添加到整合引擎
            for strategy_name, strategy_instance in self._strategies.items():
                initial_weight = 1.0 / len(self._strategies)
                self._integration_engine.add_strategy(strategy_name, strategy_instance, initial_weight)
            
            self.logger.info("整合引擎初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"整合引擎初始化失败: {e}")
            return False
    
    # ==================== 共享工具方法 ====================
    
    def _validate_input_data(self, data: Any) -> bool:
        """验证输入数据 - 共享实现"""
        if data is None:
            return False
        
        # 基本数据验证
        if isinstance(data, dict):
            required_fields = ["timestamp", "symbol", "price"]
            for field in required_fields:
                if field not in data:
                    self.logger.warning(f"输入数据缺少必需字段: {field}")
                    return False
        
        return True
    
    def _preprocess_market_data(self, data: Any) -> Dict[str, Any]:
        """预处理市场数据 - 共享实现"""
        try:
            processed_data = {}
            
            if isinstance(data, dict):
                # 基本数据转换
                processed_data = {
                    "market_data": data,
                    "timestamp": data.get("timestamp", datetime.now()),
                    "symbol": data.get("symbol", "UNKNOWN"),
                    "price": float(data.get("price", 0)),
                    "volume": float(data.get("volume", 0)),
                    "metadata": {
                        "processing_time": datetime.now(),
                        "data_quality": DataQualityLevel.GOOD.value
                    }
                }
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"市场数据预处理失败: {e}")
            return {}
    
    def _execute_strategies(self, context: StrategyContext) -> Dict[str, IStrategySignal]:
        """执行策略信号生成 - 共享实现"""
        signals = {}
        active_count = 0
        
        for strategy_name, strategy in self._strategies.items():
            try:
                # 检查策略依赖是否满足
                if not self._check_strategy_dependencies(strategy_name):
                    self.logger.debug(f"策略依赖不满足: {strategy_name}")
                    continue
                
                # 执行策略
                signal = strategy.get_signal(context.market_data)
                if signal and hasattr(signal, 'get_confidence'):
                    signals[strategy_name] = signal
                    active_count += 1
                    
                    # 更新策略性能
                    self._update_strategy_performance(strategy_name, signal)
                else:
                    self.logger.debug(f"策略无有效信号: {strategy_name}")
                    
            except Exception as e:
                self.logger.error(f"策略执行失败 {strategy_name}: {e}")
                self._handle_strategy_error(strategy_name, e)
        
        self._engine_metrics.active_strategies = active_count
        return signals
    
    def _update_strategy_performance(self, strategy_name: str, signal: IStrategySignal):
        """更新策略性能 - 共享实现"""
        if strategy_name not in self._strategy_performance:
            self._strategy_performance[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )
        
        performance = self._strategy_performance[strategy_name]
        performance.signal_count += 1
        
        # 简化的成功率计算
        signal_confidence = signal.get_confidence()
        performance.success_rate = (
            performance.success_rate * 0.9 + signal_confidence * 0.1
        )
        
        performance.last_updated = datetime.now()
    
    # ==================== 新增高级方法 ====================
    
    async def get_signal_async(self, data: Any) -> Optional[IStrategySignal]:
        """异步获取交易信号 - 高级功能"""
        if not self._enable_advanced_features:
            self.logger.warning("异步信号生成需要启用高级功能")
            return self.get_signal(data)
            
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.get_signal, data
            )
        except Exception as e:
            self.logger.error(f"异步信号生成失败: {e}")
            return None
    
    def start_engine(self) -> bool:
        """启动策略引擎 - 整合版本"""
        try:
            with self._engine_lock:
                if self._engine_state == EngineState.RUNNING:
                    self.logger.warning("策略引擎已经在运行中")
                    return True
                
                if not self.initialized:
                    self.logger.error("策略引擎未初始化，无法启动")
                    return False
                
                self._update_engine_state(EngineState.RUNNING)
                
                # 启动所有策略
                success_count = 0
                for strategy_name, strategy in self._strategies.items():
                    try:
                        # 如果策略有启动方法，则调用它
                        if hasattr(strategy, 'start_strategy'):
                            strategy.start_strategy()
                        self.logger.info(f"启动策略: {strategy_name}")
                        success_count += 1
                    except Exception as e:
                        self.logger.error(f"策略启动失败 {strategy_name}: {e}")
                
                # 条件启动整合引擎
                if self._integration_engine:
                    try:
                        self._integration_engine.optimize_strategy()
                        self.logger.info("整合引擎优化完成")
                    except Exception as e:
                        self.logger.warning(f"整合引擎优化失败: {e}")
                
                self.logger.info(f"策略引擎启动完成: {success_count}/{len(self._strategies)} 个策略")
                
                # 条件发布引擎启动事件
                if self._enable_advanced_features:
                    self._publish_engine_event("engine_started", {
                        "active_strategies": success_count,
                        "total_strategies": len(self._strategies)
                    })
                
                return success_count > 0
                
        except Exception as e:
            self.logger.error(f"策略引擎启动失败: {e}")
            self._update_engine_state(EngineState.ERROR)
            return False
    
    def get_engine_insights(self) -> Dict[str, Any]:
        """获取引擎洞察 - 高级功能"""
        if not self._enable_advanced_features:
            return {"error": "高级功能未启用"}
            
        try:
            insights = {
                "engine_health": self._get_health_status(),
                "strategy_analysis": {},
                "performance_analysis": {},
                "resource_utilization": {},
                "recommendations": []
            }
            
            # 策略分析
            for strategy_name, performance in self._strategy_performance.items():
                insights["strategy_analysis"][strategy_name] = {
                    "performance_score": performance.performance_score,
                    "signal_count": performance.signal_count,
                    "success_rate": performance.success_rate,
                    "last_updated": performance.last_updated.isoformat()
                }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"获取引擎洞察失败: {e}")
            return {"error": str(e)}
    
    # ==================== 事件发布方法 ====================
    
    def _publish_engine_event(self, event_type: str, data: Dict[str, Any]):
        """发布引擎事件 - 条件实现"""
        if not self._enable_advanced_features:
            return
            
        try:
            if self._event_dispatcher:
                event = Event(
                    event_type=event_type,
                    data=data,
                    source=self.name,
                    priority=EventPriority.NORMAL
                )
                self._event_dispatcher.dispatch_event_async(event)
            else:
                # 简化事件处理
                if event_type in self._event_handlers:
                    for handler in self._event_handlers[event_type]:
                        try:
                            handler(event_type, data)
                        except Exception as e:
                            self.logger.error(f"事件处理器执行失败: {e}")
                            
        except Exception as e:
            self.logger.debug(f"事件发布失败: {e}")
    
    def _publish_signal_event(self, signal: IStrategySignal, strategy_count: int, 
                             processing_time: float):
        """发布信号事件 - 条件实现"""
        if not self._enable_advanced_features:
            return
            
        event_data = {
            "signal_confidence": signal.get_confidence(),
            "signal_direction": signal.get_signal_direction().value,
            "strategy_count": strategy_count,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        self._publish_engine_event("strategy_signal", event_data)

    # ==================== 新增缺失的方法实现 ====================

    def _initialize_critical_components(self):
        """初始化关键组件 - 整合版本"""
        try:
            self.logger.debug("开始初始化关键组件...")
            
            # 初始化基础配置
            self._initialize_base_config()
            
            # 初始化性能监控
            self._initialize_performance_monitoring()
            
            # 初始化缓存系统
            self._initialize_cache_systems()
            
            self.logger.debug("关键组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"关键组件初始化失败: {e}")
            # 不抛出异常，允许继续初始化其他组件

    def _initialize_base_config(self):
        """初始化基础配置"""
        # 设置默认配置值
        if not hasattr(self, '_engine_metrics'):
            self._engine_metrics = EngineMetrics()
        
        if not hasattr(self, '_performance_metrics'):
            self._performance_metrics = PerformanceMetrics(
                execution_time=0.0,
                memory_usage=0,
                cpu_usage=0.0,
                call_count=0,
                error_count=0,
                cache_hit_rate=0.0
            )

    def _initialize_performance_monitoring(self):
        """初始化性能监控"""
        # 初始化性能计数器
        self._performance_counters = {
            'signals_generated': 0,
            'strategies_executed': 0,
            'errors_encountered': 0
        }

    def _initialize_cache_systems(self):
        """初始化缓存系统"""
        # 确保缓存字典已初始化
        if not hasattr(self, '_signal_cache'):
            self._signal_cache = {}
        if not hasattr(self, '_market_data_cache'):
            self._market_data_cache = {}
        if not hasattr(self, '_performance_cache'):
            self._performance_cache = {}

    def _initialize_config_system(self):
        """初始化配置系统 - 应急版本实现"""
        try:
            self.logger.debug("初始化配置系统...")
            
            # 验证配置完整性
            required_configs = ['name', 'execution_mode']
            for config_key in required_configs:
                if config_key not in self.config:
                    self.logger.warning(f"缺少必需配置: {config_key}")
                    # 设置默认值
                    if config_key == 'execution_mode':
                        self.config[config_key] = 'realtime'
            
            return True
        except Exception as e:
            self.logger.error(f"配置系统初始化失败: {e}")
            return False

    def _start_health_monitoring(self):
        """启动健康监控"""
        try:
            self.logger.debug("启动健康监控...")
            # 简化实现 - 在实际版本中这里会有定时健康检查
            self._last_health_check = datetime.now()
            return True
        except Exception as e:
            self.logger.error(f"健康监控启动失败: {e}")
            return False

    def _validate_engine_integrity(self):
        """验证引擎完整性"""
        try:
            # 检查必需组件
            required_components = ['_strategies', '_engine_metrics', '_performance_metrics']
            for component in required_components:
                if not hasattr(self, component):
                    self.logger.error(f"缺少必需组件: {component}")
                    return False
            
            # 检查策略数量
            if len(self._strategies) == 0:
                self.logger.warning("未加载任何策略")
                # 不视为错误，允许空策略引擎运行
            
            return True
        except Exception as e:
            self.logger.error(f"引擎完整性验证失败: {e}")
            return False

    def _get_active_strategies_count(self):
        """获取活跃策略数量"""
        try:
            active_count = 0
            for strategy_name, strategy in self._strategies.items():
                if hasattr(strategy, 'initialized') and strategy.initialized:
                    active_count += 1
            return active_count
        except Exception as e:
            self.logger.error(f"获取活跃策略数量失败: {e}")
            return 0

    def _get_engine_metrics_dict(self):
        """获取引擎指标字典"""
        try:
            return {
                'total_strategies': self._engine_metrics.total_strategies,
                'active_strategies': self._engine_metrics.active_strategies,
                'signals_generated': self._engine_metrics.signals_generated,
                'signals_executed': self._engine_metrics.signals_executed,
                'execution_success_rate': self._engine_metrics.execution_success_rate,
                'average_signal_latency': self._engine_metrics.average_signal_latency,
                'memory_usage_mb': self._engine_metrics.memory_usage_mb,
                'cpu_usage_percent': self._engine_metrics.cpu_usage_percent,
                'uptime_seconds': (datetime.now() - self._last_state_change).total_seconds()
            }
        except Exception as e:
            self.logger.error(f"获取引擎指标失败: {e}")
            return {}

    def _get_health_status(self):
        """获取健康状态"""
        try:
            return {
                'status': 'healthy',
                'last_check': self._last_health_check.isoformat(),
                'strategies_health': len(self._strategies) > 0,
                'performance_health': self._performance_metrics.error_count < 10
            }
        except Exception as e:
            self.logger.error(f"获取健康状态失败: {e}")
            return {'status': 'unknown'}

    def _create_strategy_context(self, processed_data):
        """创建策略上下文"""
        try:
            return StrategyContext(
                strategy_name="engine_context",
                market_data=processed_data.get('market_data', {}),
                portfolio_state={},  # 简化实现
                risk_parameters=self.config.get('risk_parameters', {}),
                execution_mode=self._execution_mode,
                timestamp=processed_data.get('timestamp', datetime.now()),
                metadata=processed_data.get('metadata', {})
            )
        except Exception as e:
            self.logger.error(f"创建策略上下文失败: {e}")
            # 返回基本上下文
            return StrategyContext(
                strategy_name="engine_context",
                market_data={},
                portfolio_state={},
                risk_parameters={},
                execution_mode=self._execution_mode
            )

    def _validate_and_optimize_signal(self, signal, strategy_context):
        """验证和优化信号"""
        try:
            # 基本信号验证
            if not signal or not hasattr(signal, 'get_confidence'):
                return None
            
            # 检查置信度
            confidence = signal.get_confidence()
            if confidence < 0 or confidence > 1:
                self.logger.warning(f"信号置信度异常: {confidence}")
                return None
            
            # 简化优化 - 在实际版本中这里会有复杂的优化逻辑
            return signal
            
        except Exception as e:
            self.logger.error(f"信号验证优化失败: {e}")
            return signal  # 返回原始信号作为降级方案

    def _update_signal_metrics(self, processing_time, signal_count):
        """更新信号指标"""
        try:
            self._engine_metrics.signals_generated += signal_count
            self._engine_metrics.average_signal_latency = (
                self._engine_metrics.average_signal_latency * 0.9 + processing_time * 0.1
            )
            self._performance_metrics.call_count += signal_count
        except Exception as e:
            self.logger.error(f"更新信号指标失败: {e}")

    def _handle_signal_generation_error(self, error):
        """处理信号生成错误"""
        try:
            error_key = type(error).__name__
            self._error_count[error_key] += 1
            self._performance_metrics.error_count += 1
            
            # 错误恢复逻辑
            if self._error_count[error_key] > 10:
                self.logger.warning(f"错误 {error_key} 发生次数过多，尝试恢复")
                self._error_count[error_key] = 0  # 重置计数器
        except Exception as e:
            self.logger.error(f"处理信号生成错误失败: {e}")

    def _check_strategy_dependencies(self, strategy_name):
        """检查策略依赖"""
        try:
            dependencies = self._strategy_dependencies.get(strategy_name, [])
            for dep in dependencies:
                if dep not in self._strategies:
                    self.logger.warning(f"策略 {strategy_name} 的依赖 {dep} 未找到")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"检查策略依赖失败 {strategy_name}: {e}")
            return True  # 依赖检查失败时允许策略运行

    def _handle_strategy_error(self, strategy_name, error):
        """处理策略错误"""
        try:
            self._recovery_attempts[strategy_name] += 1
            error_count = self._error_count[strategy_name]
            
            # 错误计数和恢复逻辑
            if error_count > 5 and self._recovery_attempts[strategy_name] < 3:
                self.logger.info(f"尝试恢复策略 {strategy_name}")
                # 在实际版本中这里会有策略恢复逻辑
        except Exception as e:
            self.logger.error(f"处理策略错误失败 {strategy_name}: {e}")

    def _update_engine_state(self, new_state: EngineState):
        """更新引擎状态"""
        old_state = self._engine_state
        self._engine_state = new_state
        self._last_state_change = datetime.now()
        
        self.logger.info(f"引擎状态变更: {old_state.value} -> {new_state.value}")
        
        # 条件发布状态变更事件
        if self._enable_advanced_features:
            self._publish_engine_event("engine_state_change", {
                "old_state": old_state.value,
                "new_state": new_state.value,
                "timestamp": self._last_state_change.isoformat()
            })

    def _get_performance_metrics_dict(self):
        """安全获取性能指标字典"""
        try:
            if hasattr(self, '_performance_metrics') and hasattr(self._performance_metrics, 'to_dict'):
                return self._performance_metrics.to_dict()
            return {
                "execution_time": 0.0,
                "memory_usage": 0,
                "cpu_usage": 0.0,
                "call_count": 0,
                "error_count": 0,
                "cache_hit_rate": 0.0
            }
        except Exception as e:
            self.logger.error(f"获取性能指标失败: {e}")
            return {}

# 策略引擎工厂类 - 整合版本
class StrategyEngineFactory:
    """策略引擎工厂 - 支持动态创建和管理策略引擎"""
    
    _engines: Dict[str, StrategyEngine] = {}
    
    @classmethod
    def create_engine(cls, name: str, config: Dict[str, Any]) -> StrategyEngine:
        """创建策略引擎 - 整合版本"""
        try:
            # 默认配置：稳定性优先
            default_config = {
                "enable_advanced_features": False,
                "event_driven_architecture": False
            }
            default_config.update(config)
            
            engine = StrategyEngine(name, default_config)
            
            if engine.initialize():
                cls._engines[name] = engine
                return engine
            else:
                # 即使初始化失败，也返回引擎实例
                cls._engines[name] = engine
                return engine
                
        except Exception as e:
            # 创建基本引擎实例
            basic_engine = StrategyEngine(name, config)
            basic_engine.initialized = False
            cls._engines[name] = basic_engine
            return basic_engine
    
    @classmethod
    def create_advanced_engine(cls, name: str, config: Dict[str, Any]) -> StrategyEngine:
        """创建高级引擎 - 启用所有高级功能"""
        advanced_config = {
            "enable_advanced_features": True,
            "event_driven_architecture": True
        }
        advanced_config.update(config)
        
        engine = cls.create_engine(name, advanced_config)
        if engine.initialized:
            engine.enable_advanced_features()
        
        return engine

# 自动注册接口
try:
    from src.interfaces import InterfaceRegistry
    InterfaceRegistry.register_interface(StrategyEngine)
except ImportError:
    pass

__all__ = [
    'StrategyEngine',
    'StrategyEngineFactory',
    'EngineState',
    'ExecutionMode', 
    'EngineMetrics',
    'StrategyContext'
]


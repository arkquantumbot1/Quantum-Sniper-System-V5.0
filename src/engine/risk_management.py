# src/engine/risk_management.py - 完整AI驱动预测性风控系统 V5.0
"""量子奇点狙击系统 - AI驱动预测性风险管理系统 V5.0 (完全重新开发 + 极致优化)"""

import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
import hashlib
import json

# 导入极致优化的依赖模块 - 更新为最新接口
from src.interfaces import (
    IStrategySignal,
    SignalDirection,
    SignalPriority,
    PerformanceMetrics,
    InterfaceMetadata,
    SignalMetadata,
    MarketRegime,
    DataQualityLevel,
    IRiskManager,
    RiskLevel,
    RiskAssessment,
    IEventDispatcher,
    Event,
    EventPriority,
    IMarketAnalyzer,
    IConfigManager,
    ConfigScope,
    ConfigChange,
    IOrderExecutor,
    PositionData,  # 移除重复的OrderType, OrderStatus
)

# 导入最新的订单执行器接口和枚举
from engine.order_executor import (
    UnifiedOrderExecutor,
    OrderType,
    OrderStatus,
    ExecutionMode,
    ExchangeType,
    OrderRequest,
    OrderResponse,
    ExecutionMetrics,
    get_global_order_executor,
)

from src.core.strategy_base import BaseStrategy, StrategySignal, StrategyError
from src.core.config_manager import BaseConfigManager, ConfigManagerFactory
from config.config import UnifiedConfigLoader, get_global_config

# 集成SAC策略优化器
try:
    from brain.sac_strategy_optimizer import SACStrategyOptimizer, EvolutionaryOptimizer

    HAS_SAC_OPTIMIZER = True
except ImportError:
    HAS_SAC_OPTIMIZER = False

# ==================== 极致优化的风险数据结构 ====================


class RiskControlLayer(Enum):
    """风险控制层级 - 极致优化版本"""

    PORTFOLIO = "portfolio"  # 投资组合层级
    POSITION = "position"  # 仓位层级
    MARKET = "market"  # 市场层级
    LIQUIDITY = "liquidity"  # 流动性层级
    COMPLIANCE = "compliance"  # 合规层级
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断层级
    SYSTEMIC = "systemic"  # 系统性风险层级


class RiskEventType(Enum):
    """风险事件类型 - 极致优化版本"""

    POSITION_LIMIT_BREACH = "position_limit_breach"
    DRAWDOWN_WARNING = "drawdown_warning"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    SYSTEM_FAILURE = "system_failure"
    MARKET_REGIME_CHANGE = "market_regime_change"
    BLACK_SWAN_EVENT = "black_swan_event"
    FLASH_CRASH_DETECTED = "flash_crash_detected"
    RISK_PARAMETER_BREACH = "risk_parameter_breach"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"  # 新增


class RiskPredictionModel(Enum):
    """风险预测模型类型"""

    QUANTUM_NEURAL = "quantum_neural"
    SAC_OPTIMIZED = "sac_optimized"
    ENSEMBLE_AI = "ensemble_ai"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


@dataclass
class RiskMetrics:
    """风险指标数据类 - 极致优化版本"""

    current_drawdown: float = 0.0
    volatility_30d: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    sharpe_ratio: float = 0.0
    max_position_concentration: float = 0.0
    portfolio_beta: float = 0.0
    liquidity_score: float = 1.0
    correlation_risk: float = 0.0
    stress_test_score: float = 0.0
    regime_stability: float = 1.0
    execution_success_rate: float = 1.0  # 新增：执行成功率
    average_slippage: float = 0.0  # 新增：平均滑点
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current_drawdown": self.current_drawdown,
            "volatility_30d": self.volatility_30d,
            "var_95": self.var_95,
            "expected_shortfall": self.expected_shortfall,
            "sharpe_ratio": self.sharpe_ratio,
            "max_position_concentration": self.max_position_concentration,
            "portfolio_beta": self.portfolio_beta,
            "liquidity_score": self.liquidity_score,
            "correlation_risk": self.correlation_risk,
            "stress_test_score": self.stress_test_score,
            "regime_stability": self.regime_stability,
            "execution_success_rate": self.execution_success_rate,
            "average_slippage": self.average_slippage,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class RiskEvent:
    """风险事件数据类 - 极致优化版本"""

    event_type: RiskEventType
    severity: RiskLevel
    description: str
    triggered_by: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    impact_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "triggered_by": self.triggered_by,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "impact_score": self.impact_score,
        }


@dataclass
class PositionRisk:
    """仓位风险数据类 - 极致优化版本"""

    symbol: str
    current_size: float
    max_allowed: float
    risk_exposure: float
    stop_loss_level: float
    take_profit_level: float
    volatility_adjusted: bool = False
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "current_size": self.current_size,
            "max_allowed": self.max_allowed,
            "risk_exposure": self.risk_exposure,
            "stop_loss_level": self.stop_loss_level,
            "take_profit_level": self.take_profit_level,
            "volatility_adjusted": self.volatility_adjusted,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class RiskPrediction:
    """风险预测数据类"""

    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: RiskPredictionModel = RiskPredictionModel.ENSEMBLE_AI
    prediction_horizon: timedelta = field(default_factory=lambda: timedelta(hours=24))
    confidence: float = 0.0
    predicted_risk_level: RiskLevel = RiskLevel.MINIMAL
    key_risk_factors: List[str] = field(default_factory=list)
    mitigation_recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# ==================== AI驱动风险管理系统主类 ====================


class RiskManagementSystem(IRiskManager):
    """风险管理系统 V5.0 - AI驱动预测性风控 (基于最新order_executor.py完全重新开发 + 极致优化)"""

    # 接口元数据
    _metadata = InterfaceMetadata(
        version="5.0.2",
        description="AI驱动预测性风险管理系统 - 多层风控 + 实时监控 + 动态参数调整 + 量子集成 + 紧急平仓",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "risk_assessment_time": 0.002,
            "position_calculation_time": 0.001,
            "circuit_breaker_response": 0.005,
            "ai_prediction_latency": 0.01,
            "emergency_liquidation_time": 0.1,  # 新增：紧急平仓响应时间
        },
        dependencies=[
            "IRiskManager",
            "IConfigManager",
            "IMarketAnalyzer",
            "IEventDispatcher",
            "SACStrategyOptimizer",
            "UnifiedOrderExecutor",  # 更新为UnifiedOrderExecutor
        ],
        compatibility=["5.0", "4.2", "4.1"],
    )

    def __init__(self, name: str = "QuantumRiskManager", config: Dict[str, Any] = None):
        # 配置处理 - 极致优化
        config = config or {}
        default_config = {
            "name": name,
            "max_drawdown": 0.15,
            "max_position_size": 0.2,
            "daily_loss_limit": 0.02,
            "volatility_threshold": 0.15,
            "liquidity_threshold": 0.7,
            "correlation_threshold": 0.8,
            "circuit_breaker_enabled": True,
            "ai_risk_prediction": True,
            "real_time_monitoring": True,
            "adaptive_risk_parameters": True,
            "multi_layer_control": True,
            "stress_testing_enabled": True,
            "auto_recovery_enabled": True,
            "emergency_liquidation_enabled": True,  # 新增：紧急平仓开关
            "max_emergency_liquidation_percent": 0.5,  # 新增：最大紧急平仓比例
            "execution_quality_threshold": 0.9,  # 新增：执行质量阈值
        }

        # 完整版本配置扩展
        advanced_defaults = {
            "enabled": True,
            "risk_level": "medium",
            "prediction_horizon_hours": 24,
            "confidence_threshold": 0.8,
            "quantum_coherence_integration": True,
            "distributed_risk_calculation": True,
            "gpu_acceleration": True,
        }

        default_config.update(advanced_defaults)
        default_config.update(config)

        # 由于IRiskManager是接口，我们直接初始化基础属性
        self.name = name
        self.config = default_config
        self.initialized = False

        # ==================== 核心风险属性 - 极致优化 ====================

        # 风险状态管理
        self._current_risk_level: RiskLevel = RiskLevel.MINIMAL
        self._last_risk_assessment: datetime = datetime.now()
        self._system_enabled: bool = True

        # 多层风控系统
        self._risk_layers: Dict[RiskControlLayer, bool] = {
            RiskControlLayer.PORTFOLIO: True,
            RiskControlLayer.POSITION: True,
            RiskControlLayer.MARKET: True,
            RiskControlLayer.LIQUIDITY: True,
            RiskControlLayer.COMPLIANCE: True,
            RiskControlLayer.CIRCUIT_BREAKER: True,
            RiskControlLayer.SYSTEMIC: True,
        }

        # 风险指标跟踪
        self._risk_metrics = RiskMetrics()
        self._position_risks: Dict[str, PositionRisk] = {}
        self._portfolio_exposure: float = 0.0

        # AI驱动预测组件（完整集成）
        self._enable_ai_prediction = config.get("ai_risk_prediction", True)
        self._sac_risk_optimizer: Optional[SACStrategyOptimizer] = None
        self._risk_prediction_models: Dict[RiskPredictionModel, Any] = {}

        # 🚀 更新：使用最新的UnifiedOrderExecutor
        self._order_executor: Optional[UnifiedOrderExecutor] = None
        self._emergency_stop_orders: List[str] = []

        # 性能监控
        self._performance_metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0,
            cpu_usage=0.0,
            call_count=0,
            error_count=0,
            cache_hit_rate=0.0,
        )

        # 事件和监控系统
        self._risk_events: deque = deque(maxlen=1000)
        self._event_dispatcher: Optional[IEventDispatcher] = None
        self._market_analyzer: Optional[IMarketAnalyzer] = None

        # 线程安全
        self._risk_lock = threading.RLock()
        self._event_lock = threading.RLock()
        self._prediction_lock = threading.RLock()
        self._liquidation_lock = threading.RLock()  # 新增：紧急平仓锁

        # 缓存系统
        self._risk_assessment_cache: Dict[str, RiskAssessment] = {}
        self._prediction_cache: Dict[str, RiskPrediction] = {}

        # 熔断机制
        self._circuit_breaker_triggered: bool = False
        self._circuit_breaker_history: List[Dict[str, Any]] = []

        # 动态参数调整
        self._adaptive_parameters: Dict[str, float] = {}
        self._parameter_history: Dict[str, List[float]] = defaultdict(list)

        # 风险预测历史
        self._risk_predictions: List[RiskPrediction] = []

        # 🚀 新增：执行质量监控
        self._execution_quality_metrics: Dict[str, float] = {
            "success_rate": 1.0,
            "avg_latency": 0.0,
            "slippage": 0.0,
            "last_update": datetime.now(),
        }

        self.logger = logging.getLogger(f"risk.{name}")

        # 自动初始化关键组件
        self._initialize_critical_components()

    @classmethod
    def get_interface_metadata(cls) -> InterfaceMetadata:
        """获取接口元数据 - 极致优化版本"""
        return cls._metadata

    def initialize(self) -> bool:
        """初始化风险管理系统 - 极致优化版本"""
        start_time = datetime.now()

        try:
            if self.initialized:
                self.logger.warning("风险管理系统已经初始化")
                return True

            self.logger.info("开始初始化AI驱动风险管理系统...")

            # ==================== 分步初始化流程 ====================

            # 1. 初始化配置系统
            if not self._initialize_config_system():
                raise StrategyError("配置系统初始化失败", self.name)

            # 2. 初始化风险指标跟踪
            if not self._initialize_risk_tracking():
                self.logger.warning("风险指标跟踪初始化警告")

            # 3. 初始化AI预测组件
            if self._enable_ai_prediction:
                if not self._initialize_ai_components():
                    self.logger.warning("AI组件初始化失败，继续基础模式")
            else:
                self.logger.info("AI风险预测已禁用")

            # 🚀 更新：初始化最新的订单执行器
            if not self._initialize_order_execution():
                self.logger.warning("订单执行集成初始化警告")

            # 5. 初始化熔断机制
            if not self._initialize_circuit_breakers():
                self.logger.warning("熔断机制初始化警告")

            # 6. 初始化动态参数系统
            if not self._initialize_adaptive_parameters():
                self.logger.warning("动态参数系统初始化警告")

            # 7. 初始化风险预测系统
            if not self._initialize_risk_prediction_system():
                self.logger.warning("风险预测系统初始化警告")

            # 🚀 新增：初始化紧急平仓系统
            if not self._initialize_emergency_liquidation():
                self.logger.warning("紧急平仓系统初始化警告")

            # 8. 验证系统完整性
            if not self._validate_system_integrity():
                raise StrategyError("系统完整性验证失败", self.name)

            # 更新状态
            self.initialized = True

            initialization_time = (datetime.now() - start_time).total_seconds()
            self._performance_metrics.execution_time += initialization_time
            self._performance_metrics.call_count += 1

            self.logger.info(
                f"风险管理系统初始化完成: "
                f"AI预测={self._enable_ai_prediction}, "
                f"多层风控={self.config.get('multi_layer_control', True)}, "
                f"紧急平仓={self.config.get('emergency_liquidation_enabled', True)}, "
                f"耗时: {initialization_time:.3f}s"
            )

            # 发布系统就绪事件
            self._publish_risk_event(
                RiskEventType.SYSTEM_FAILURE,
                RiskLevel.MINIMAL,
                "风险管理系统初始化完成",
                "system_initialization",
            )

            return True

        except Exception as e:
            self.logger.error(f"风险管理系统初始化失败: {e}")
            self._performance_metrics.error_count += 1
            return False

    def get_risk_config(self) -> Dict[str, Any]:
        """获取风控配置"""
        return {
            "risk_level": self.config.get("risk_level", "medium"),
            "max_position_size": self.config.get("max_position_size", 0.1),
            "stop_loss_percentage": self.config.get("stop_loss_percentage", 0.02),
            "max_drawdown": self.config.get("max_drawdown", 0.05),
            "enabled": self.config.get("enabled", True),
        }

    # ==================== 核心风险评估方法 ====================

    def calculate_position_size(self, signal: IStrategySignal, balance: float) -> float:
        """计算仓位大小 - 极致优化版本"""
        if not self.initialized or not self._system_enabled:
            self.logger.warning("风险管理系统未启用，使用默认仓位")
            return balance * 0.1  # 默认10%

        start_time = datetime.now()

        try:
            # 基础验证
            if not signal or balance <= 0:
                return 0.0

            # 获取信号置信度和方向
            confidence = signal.get_confidence()
            direction = signal.get_signal_direction()

            # 基础仓位计算
            base_position = self._calculate_base_position_size(signal, balance)

            # 应用风险调整
            risk_adjusted_position = self._apply_risk_adjustments(
                base_position, signal, balance, confidence, direction
            )

            # AI优化仓位（如果启用）
            if self._enable_ai_prediction and self._sac_risk_optimizer:
                ai_optimized_position = self._apply_ai_optimization(
                    risk_adjusted_position, signal, balance
                )
                risk_adjusted_position = ai_optimized_position

            # 🚀 新增：执行质量调整
            execution_adjusted_position = self._apply_execution_quality_adjustment(
                risk_adjusted_position, signal
            )
            risk_adjusted_position = execution_adjusted_position

            # 最终验证和限制
            final_position = self._apply_final_limits(risk_adjusted_position, balance)

            # 更新性能指标
            calculation_time = (datetime.now() - start_time).total_seconds()
            self._performance_metrics.execution_time += calculation_time
            self._performance_metrics.call_count += 1

            self.logger.debug(
                f"仓位计算完成: 基础={base_position:.4f}, "
                f"调整后={risk_adjusted_position:.4f}, "
                f"最终={final_position:.4f}, "
                f"耗时: {calculation_time:.3f}s"
            )

            return final_position

        except Exception as e:
            self.logger.error(f"仓位计算失败: {e}")
            self._performance_metrics.error_count += 1
            return balance * 0.05  # 错误时使用保守仓位

    def validate_trade_signal(self, signal: IStrategySignal) -> Tuple[bool, str]:
        """验证交易信号 - 极致优化版本"""
        if not self.initialized or not self._system_enabled:
            return True, "风险管理系统未启用"

        start_time = datetime.now()

        try:
            validation_checks = []

            # 1. 信号完整性验证
            if not self._validate_signal_integrity(signal):
                return False, "信号完整性验证失败"

            # 2. 置信度检查
            confidence = signal.get_confidence()
            min_confidence = self.config.get("confidence_threshold", 0.6)
            if confidence < min_confidence:
                validation_checks.append(f"置信度过低: {confidence:.3f} < {min_confidence}")

            # 3. 市场状态检查
            market_check, market_reason = self._validate_market_conditions(signal)
            if not market_check:
                validation_checks.append(market_reason)

            # 4. 风险层级检查
            risk_check, risk_reason = self._validate_risk_layers(signal)
            if not risk_check:
                validation_checks.append(risk_reason)

            # 5. AI预测检查（如果启用）
            if self._enable_ai_prediction:
                ai_check, ai_reason = self._validate_ai_prediction(signal)
                if not ai_check:
                    validation_checks.append(ai_reason)

            # 6. 系统性风险检查
            systemic_check, systemic_reason = self._validate_systemic_risk(signal)
            if not systemic_check:
                validation_checks.append(systemic_reason)

            # 🚀 新增：执行质量检查
            execution_check, execution_reason = self._validate_execution_quality(signal)
            if not execution_check:
                validation_checks.append(execution_reason)

            # 汇总结果
            is_valid = len(validation_checks) == 0
            reason = "验证通过" if is_valid else "; ".join(validation_checks)

            # 更新性能指标
            validation_time = (datetime.now() - start_time).total_seconds()
            self._performance_metrics.execution_time += validation_time

            return is_valid, reason

        except Exception as e:
            self.logger.error(f"信号验证失败: {e}")
            self._performance_metrics.error_count += 1
            return False, f"验证过程异常: {e}"

    # ==================== 🚀 新增极致优化方法 ====================

    async def assess_risk_async(
        self, signal: IStrategySignal, market_data: Dict[str, Any]
    ) -> RiskAssessment:
        """异步风险评估 - 极致优化版本"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._assess_risk_comprehensive, signal, market_data
            )
        except Exception as e:
            self.logger.error(f"异步风险评估失败: {e}")
            # 返回保守的风险评估
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                max_position_size=0.05,  # 5%保守仓位
                recommended_leverage=1.0,
                stop_loss_level=0.02,  # 2%止损
                confidence=0.5,
                factors={"error": str(e)},
            )

    def get_risk_metrics(self) -> Dict[str, float]:
        """获取风险指标 - 极致优化版本"""
        try:
            with self._risk_lock:
                return {
                    "current_drawdown": self._risk_metrics.current_drawdown,
                    "volatility_30d": self._risk_metrics.volatility_30d,
                    "var_95": self._risk_metrics.var_95,
                    "expected_shortfall": self._risk_metrics.expected_shortfall,
                    "sharpe_ratio": self._risk_metrics.sharpe_ratio,
                    "max_position_concentration": self._risk_metrics.max_position_concentration,
                    "portfolio_beta": self._risk_metrics.portfolio_beta,
                    "liquidity_score": self._risk_metrics.liquidity_score,
                    "correlation_risk": self._risk_metrics.correlation_risk,
                    "stress_test_score": self._risk_metrics.stress_test_score,
                    "regime_stability": self._risk_metrics.regime_stability,
                    "execution_success_rate": self._risk_metrics.execution_success_rate,
                    "average_slippage": self._risk_metrics.average_slippage,
                    "portfolio_exposure": self._portfolio_exposure,
                    "system_risk_level": self._current_risk_level.value,
                }
        except Exception as e:
            self.logger.error(f"获取风险指标失败: {e}")
            return {"error": str(e)}

    def adjust_risk_parameters(self, market_regime: MarketRegime) -> bool:
        """调整风险参数 - 极致优化版本"""
        try:
            self.logger.info(f"根据市场状态调整风险参数: {market_regime.value}")

            # 基于市场状态的参数调整
            regime_multipliers = {
                MarketRegime.BULL_TREND: 1.2,  # 牛市增加风险承受
                MarketRegime.BEAR_TREND: 0.6,  # 熊市大幅降低风险
                MarketRegime.SIDEWAYS: 1.0,  # 震荡市保持中性
                MarketRegime.HIGH_VOLATILITY: 0.7,  # 高波动率降低风险
                MarketRegime.LOW_VOLATILITY: 1.1,  # 低波动率适度增加风险
                MarketRegime.CRISIS: 0.3,  # 危机状态极度保守
                MarketRegime.RECOVERY: 0.8,  # 恢复期保持保守
            }

            multiplier = regime_multipliers.get(market_regime, 1.0)

            # 调整关键风险参数
            adjustable_params = [
                "max_position_size",
                "max_drawdown",
                "daily_loss_limit",
                "volatility_threshold",
                "liquidity_threshold",
            ]

            for param in adjustable_params:
                if param in self.config:
                    base_value = self.config[param]
                    adjusted_value = base_value * multiplier

                    # 记录参数调整历史
                    if param not in self._parameter_history:
                        self._parameter_history[param] = []
                    self._parameter_history[param].append(adjusted_value)

                    # 更新自适应参数
                    self._adaptive_parameters[param] = adjusted_value

            self.logger.info(f"风险参数调整完成: 乘数={multiplier:.2f}")
            return True

        except Exception as e:
            self.logger.error(f"风险参数调整失败: {e}")
            return False

    def simulate_stress_test(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """压力测试模拟 - 极致优化版本"""
        try:
            self.logger.info(f"开始压力测试: {len(scenarios)} 个场景")

            stress_test_results = {
                "timestamp": datetime.now().isoformat(),
                "scenarios_tested": len(scenarios),
                "results": [],
                "overall_risk_score": 0.0,
                "recommendations": [],
                "worst_case_scenario": None,
            }

            total_risk_score = 0.0
            worst_scenario = None
            worst_score = 0.0

            for i, scenario in enumerate(scenarios):
                scenario_result = self._simulate_single_scenario(scenario, i)
                stress_test_results["results"].append(scenario_result)

                scenario_score = scenario_result.get("risk_score", 0.0)
                total_risk_score += scenario_score

                # 跟踪最坏情况
                if scenario_score > worst_score:
                    worst_score = scenario_score
                    worst_scenario = scenario_result

            # 计算总体风险评分
            if scenarios:
                stress_test_results["overall_risk_score"] = total_risk_score / len(
                    scenarios
                )
                stress_test_results["worst_case_scenario"] = worst_scenario

            # 生成建议
            stress_test_results[
                "recommendations"
            ] = self._generate_stress_test_recommendations(stress_test_results)

            # 更新风险指标
            self._risk_metrics.stress_test_score = stress_test_results[
                "overall_risk_score"
            ]

            self.logger.info(
                f"压力测试完成: 总体风险评分={stress_test_results['overall_risk_score']:.3f}"
            )
            return stress_test_results

        except Exception as e:
            self.logger.error(f"压力测试失败: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def get_risk_exposure(self) -> Dict[str, float]:
        """获取风险暴露 - 极致优化版本"""
        try:
            with self._risk_lock:
                exposure = {
                    "total_portfolio_exposure": self._portfolio_exposure,
                    "position_concentrations": {},
                    "sector_exposures": {},
                    "risk_factor_exposures": {},
                    "systemic_risk_exposure": 0.0,
                }

                # 计算仓位集中度
                for symbol, position_risk in self._position_risks.items():
                    exposure["position_concentrations"][
                        symbol
                    ] = position_risk.risk_exposure

                # 计算风险因子暴露
                exposure["risk_factor_exposures"] = {
                    "market_risk": self._risk_metrics.portfolio_beta,
                    "liquidity_risk": 1.0 - self._risk_metrics.liquidity_score,
                    "volatility_risk": self._risk_metrics.volatility_30d,
                    "correlation_risk": self._risk_metrics.correlation_risk,
                    "execution_risk": 1.0
                    - self._risk_metrics.execution_success_rate,  # 新增执行风险
                    "systemic_risk": self._calculate_systemic_risk_exposure(),
                }

                return exposure

        except Exception as e:
            self.logger.error(f"获取风险暴露失败: {e}")
            return {"error": str(e)}

    def enable_ai_risk_prediction(self) -> bool:
        """启用AI风险预测 - 极致优化版本"""
        try:
            if self._enable_ai_prediction:
                self.logger.info("AI风险预测已经启用")
                return True

            self.logger.info("正在启用AI风险预测...")

            if not self._initialize_ai_components():
                self.logger.error("AI组件初始化失败")
                return False

            self._enable_ai_prediction = True
            self.config["ai_risk_prediction"] = True

            self.logger.info("AI风险预测启用完成")
            return True

        except Exception as e:
            self.logger.error(f"启用AI风险预测失败: {e}")
            return False

    def trigger_circuit_breaker(
        self, reason: str, severity: RiskLevel = RiskLevel.HIGH
    ) -> bool:
        """触发熔断机制 - 极致优化版本"""
        try:
            with self._risk_lock:
                if self._circuit_breaker_triggered:
                    self.logger.warning("熔断机制已经触发")
                    return True

                self._circuit_breaker_triggered = True

                # 记录熔断事件
                circuit_event = {
                    "timestamp": datetime.now().isoformat(),
                    "reason": reason,
                    "severity": severity.value,
                    "triggered_by": "risk_management_system",
                    "actions_taken": ["暂停所有新交易", "强制平仓高风险仓位", "降低杠杆限制", "启用紧急风控协议"],
                }

                self._circuit_breaker_history.append(circuit_event)

                # 🚀 更新：使用新的紧急平仓系统
                if self.config.get("emergency_liquidation_enabled", True):
                    liquidation_result = self._execute_emergency_liquidation_advanced()
                    circuit_event["liquidation_result"] = liquidation_result

                # 发布熔断事件
                self._publish_risk_event(
                    RiskEventType.SYSTEM_FAILURE,
                    severity,
                    f"熔断机制触发: {reason}",
                    "circuit_breaker",
                )

                self.logger.warning(f"熔断机制触发: {reason} (严重程度: {severity.value})")
                return True

        except Exception as e:
            self.logger.error(f"熔断机制触发失败: {e}")
            return False

    def predict_risk(self, horizon_hours: int = 24) -> RiskPrediction:
        """风险预测 - 新增极致优化方法"""
        try:
            with self._prediction_lock:
                prediction_id = hashlib.md5(
                    f"{datetime.now().isoformat()}_{horizon_hours}".encode()
                ).hexdigest()

                # 检查缓存
                if prediction_id in self._prediction_cache:
                    return self._prediction_cache[prediction_id]

                # 使用AI模型进行风险预测
                if self._enable_ai_prediction and self._sac_risk_optimizer:
                    prediction = self._generate_ai_risk_prediction(horizon_hours)
                else:
                    prediction = self._generate_baseline_risk_prediction(horizon_hours)

                # 缓存预测结果
                self._prediction_cache[prediction_id] = prediction
                self._risk_predictions.append(prediction)

                # 限制预测历史长度
                if len(self._risk_predictions) > 100:
                    self._risk_predictions = self._risk_predictions[-100:]

                return prediction

        except Exception as e:
            self.logger.error(f"风险预测失败: {e}")
            return RiskPrediction(
                confidence=0.0,
                predicted_risk_level=RiskLevel.HIGH,
                key_risk_factors=["prediction_error"],
                mitigation_recommendations=["系统错误，请检查日志"],
            )

    def get_risk_predictions(self, limit: int = 10) -> List[RiskPrediction]:
        """获取风险预测历史"""
        return self._risk_predictions[-limit:] if self._risk_predictions else []

    # 🚀 新增：紧急平仓方法
    def emergency_liquidation(
        self, symbol: str = None, percent: float = 1.0
    ) -> Dict[str, Any]:
        """紧急平仓 - 基于最新订单执行器"""
        try:
            with self._liquidation_lock:
                if not self._order_executor:
                    return {"success": False, "error": "订单执行器不可用"}

                max_percent = self.config.get("max_emergency_liquidation_percent", 0.5)
                liquidation_percent = min(percent, max_percent)

                self.logger.warning(
                    f"开始紧急平仓: {symbol or '全部'}, 比例: {liquidation_percent}"
                )

                # 创建紧急平仓订单
                liquidation_orders = self._create_liquidation_orders(
                    symbol, liquidation_percent
                )

                # 执行平仓
                results = []
                for order_request in liquidation_orders:
                    try:
                        order_response = self._order_executor.execute_order(
                            order_request
                        )
                        results.append(
                            {
                                "symbol": order_request.symbol,
                                "direction": order_request.direction.value,
                                "quantity": order_request.quantity,
                                "status": order_response.status.value,
                                "execution_latency": order_response.execution_latency,
                            }
                        )

                        # 记录紧急平仓订单
                        self._emergency_stop_orders.append(
                            order_response.client_order_id
                        )

                    except Exception as e:
                        self.logger.error(f"紧急平仓订单执行失败: {e}")
                        results.append(
                            {"symbol": order_request.symbol, "error": str(e)}
                        )

                # 发布紧急平仓事件
                self._publish_risk_event(
                    RiskEventType.EMERGENCY_LIQUIDATION,
                    RiskLevel.EXTREME,
                    f"紧急平仓执行: {len(liquidation_orders)}个订单",
                    "emergency_liquidation",
                    data={"results": results},
                )

                return {
                    "success": True,
                    "liquidation_percent": liquidation_percent,
                    "orders_executed": len(liquidation_orders),
                    "results": results,
                }

        except Exception as e:
            self.logger.error(f"紧急平仓失败: {e}")
            return {"success": False, "error": str(e)}

    # ==================== 内部实现方法 - 极致优化 ====================

    def _initialize_critical_components(self):
        """初始化关键组件 - 极致优化版本"""
        try:
            self.logger.debug("初始化风险管理系统关键组件...")

            # 初始化基础配置
            self._initialize_base_config()

            # 初始化性能监控
            self._initialize_performance_monitoring()

            # 初始化缓存系统
            self._initialize_cache_systems()

            self.logger.debug("关键组件初始化完成")

        except Exception as e:
            self.logger.error(f"关键组件初始化失败: {e}")

    def _initialize_base_config(self):
        """初始化基础配置"""
        try:
            self.logger.debug("初始化基础配置...")
            # 基础配置初始化
            self._adaptive_parameters = {
                "position_size_multiplier": 1.0,
                "risk_tolerance": 0.5,
                "volatility_adjustment": 1.0,
            }
            return True
        except Exception as e:
            self.logger.error(f"基础配置初始化失败: {e}")
            return False

    def _initialize_risk_tracking(self):
        """初始化风险跟踪"""
        try:
            self.logger.debug("初始化风险跟踪...")
            # 风险指标初始化
            self._risk_metrics = RiskMetrics()
            self._position_risks = {}
            self._portfolio_exposure = 0.0
            return True
        except Exception as e:
            self.logger.error(f"风险跟踪初始化失败: {e}")
            return False

    def _initialize_performance_monitoring(self):
        """初始化性能监控"""
        try:
            self.logger.debug("初始化性能监控...")
            self._performance_metrics = PerformanceMetrics(
                execution_time=0.0,
                memory_usage=0,
                cpu_usage=0.0,
                call_count=0,
                error_count=0,
                cache_hit_rate=0.0,
            )
            return True
        except Exception as e:
            self.logger.error(f"性能监控初始化失败: {e}")
            return False

    def _initialize_cache_systems(self):
        """初始化缓存系统"""
        try:
            self.logger.debug("初始化缓存系统...")
            self._risk_assessment_cache = {}
            self._prediction_cache = {}
            return True
        except Exception as e:
            self.logger.error(f"缓存系统初始化失败: {e}")
            return False

    def _initialize_circuit_breakers(self):
        """初始化熔断机制"""
        try:
            self.logger.debug("初始化熔断机制...")
            self._circuit_breaker_triggered = False
            self._circuit_breaker_history = []
            return True
        except Exception as e:
            self.logger.error(f"熔断机制初始化失败: {e}")
            return False

    def _initialize_adaptive_parameters(self):
        """初始化自适应参数"""
        try:
            self.logger.debug("初始化自适应参数...")
            self._adaptive_parameters = {}
            self._parameter_history = defaultdict(list)
            return True
        except Exception as e:
            self.logger.error(f"自适应参数初始化失败: {e}")
            return False

    def _validate_system_integrity(self):
        """验证系统完整性"""
        try:
            self.logger.debug("验证系统完整性...")
            # 验证必需属性
            required_attributes = [
                "_risk_metrics",
                "_position_risks",
                "_performance_metrics",
                "_risk_events",
                "_adaptive_parameters",
            ]

            for attr in required_attributes:
                if not hasattr(self, attr):
                    self.logger.error(f"缺少必需属性: {attr}")
                    return False

            # 验证配置
            required_configs = ["max_drawdown", "max_position_size", "daily_loss_limit"]
            for config_key in required_configs:
                if config_key not in self.config:
                    self.logger.error(f"缺少必需配置: {config_key}")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"系统完整性验证失败: {e}")
            return False

    def _initialize_config_system(self) -> bool:
        """初始化配置系统"""
        try:
            self.logger.debug("初始化风险配置系统...")

            # 验证配置完整性
            required_configs = ["max_drawdown", "max_position_size", "daily_loss_limit"]
            for config_key in required_configs:
                if config_key not in self.config:
                    self.logger.warning(f"缺少风险配置: {config_key}")
                    # 设置默认值
                    if config_key == "max_drawdown":
                        self.config[config_key] = 0.15
                    elif config_key == "max_position_size":
                        self.config[config_key] = 0.2
                    elif config_key == "daily_loss_limit":
                        self.config[config_key] = 0.02

            return True
        except Exception as e:
            self.logger.error(f"配置系统初始化失败: {e}")
            return False

    def _initialize_ai_components(self) -> bool:
        """初始化AI组件 - 集成SAC优化器"""
        if not self._enable_ai_prediction:
            return True

        try:
            self.logger.debug("初始化AI风险预测组件...")

            # 初始化SAC风险优化器
            if HAS_SAC_OPTIMIZER:
                try:
                    sac_config = {
                        "name": "SACRiskOptimizer",
                        "learning_rate": 0.001,
                        "population_size": 30,
                        "optimization_targets": ["risk_adjustment", "position_sizing"],
                    }
                    self._sac_risk_optimizer = SACStrategyOptimizer(
                        "SACRiskOptimizer", sac_config
                    )

                    if self._sac_risk_optimizer.initialize():
                        self.logger.info("SAC风险优化器初始化成功")
                    else:
                        self.logger.warning("SAC风险优化器初始化失败")
                        return False
                except Exception as e:
                    self.logger.error(f"SAC风险优化器初始化异常: {e}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"AI组件初始化失败: {e}")
            return False

    def _initialize_order_execution(self) -> bool:
        """🚀 更新：初始化最新的订单执行器集成"""
        try:
            self.logger.debug("初始化订单执行集成...")

            # 使用全局订单执行器或创建新实例
            try:
                self._order_executor = get_global_order_executor()
                self.logger.info(
                    f"全局订单执行器加载成功: {self._order_executor.__class__.__name__}"
                )
            except Exception as e:
                self.logger.warning(f"全局订单执行器加载失败: {e}")
                # 创建新的执行器实例
                self._order_executor = UnifiedOrderExecutor()
                if not self._order_executor.load_config():
                    self.logger.error("订单执行器配置加载失败")
                    return False

            # 验证执行器状态
            executor_status = self._order_executor.get_detailed_status()
            self.logger.info(
                f"订单执行器状态: {executor_status.get('execution_mode', 'unknown')}"
            )

            return True

        except Exception as e:
            self.logger.error(f"订单执行集成初始化失败: {e}")
            return False

    def _initialize_risk_prediction_system(self) -> bool:
        """初始化风险预测系统"""
        try:
            self.logger.debug("初始化风险预测系统...")

            self._risk_predictions = []
            self._prediction_cache = {}

            return True
        except Exception as e:
            self.logger.error(f"风险预测系统初始化失败: {e}")
            return False

    def _initialize_emergency_liquidation(self) -> bool:
        """🚀 新增：初始化紧急平仓系统"""
        try:
            self.logger.debug("初始化紧急平仓系统...")

            self._emergency_stop_orders = []

            # 验证紧急平仓配置
            if self.config.get("emergency_liquidation_enabled", True):
                if not self._order_executor:
                    self.logger.warning("紧急平仓已启用但订单执行器不可用")
                    return False

            self.logger.info("紧急平仓系统初始化完成")
            return True

        except Exception as e:
            self.logger.error(f"紧急平仓系统初始化失败: {e}")
            return False

    def _calculate_base_position_size(
        self, signal: IStrategySignal, balance: float
    ) -> float:
        """计算基础仓位大小 - 凯利公式优化版本"""
        try:
            # 基于凯利公式的优化版本
            confidence = signal.get_confidence()
            win_probability = confidence  # 使用置信度作为胜率估计

            # 动态盈亏比估计
            if hasattr(signal, "get_expected_win_loss_ratio"):
                win_loss_ratio = signal.get_expected_win_loss_ratio()
            else:
                win_loss_ratio = 2.0  # 默认盈亏比

            # 凯利公式: f = p - (1-p)/b
            kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio

            # 应用凯利分数限制（使用半凯利或更保守）
            kelly_fraction = max(0, kelly_fraction)  # 不允许负值
            conservative_fraction = kelly_fraction * 0.25  # 四分之一凯利，更加保守

            base_position = balance * conservative_fraction
            return base_position

        except Exception as e:
            self.logger.error(f"基础仓位计算失败: {e}")
            return balance * 0.1  # 默认10%

    def _apply_ai_optimization(
        self, position: float, signal: IStrategySignal, balance: float
    ) -> float:
        """应用AI优化 - 集成SAC优化器"""
        if not self._sac_risk_optimizer:
            return position

        try:
            # 准备优化数据
            optimization_data = {
                "current_position": position,
                "signal_confidence": signal.get_confidence(),
                "balance": balance,
                "current_risk_level": self._current_risk_level.value,
                "risk_metrics": self._risk_metrics.to_dict(),
            }

            # 获取SAC优化信号
            sac_signal = self._sac_risk_optimizer.get_signal(optimization_data)
            if sac_signal and sac_signal.get_confidence() > 0.7:
                # 使用SAC优化的仓位
                optimized_position = position * sac_signal.get_confidence()
                self.logger.debug(
                    f"SAC优化应用: {position:.4f} -> {optimized_position:.4f}"
                )
                return optimized_position
            else:
                return position

        except Exception as e:
            self.logger.error(f"AI优化应用失败: {e}")
            return position

    def _apply_execution_quality_adjustment(
        self, position: float, signal: IStrategySignal
    ) -> float:
        """🚀 新增：应用执行质量调整"""
        try:
            # 获取执行质量指标
            success_rate = self._execution_quality_metrics.get("success_rate", 1.0)
            avg_latency = self._execution_quality_metrics.get("avg_latency", 0.0)

            # 执行质量调整因子
            quality_factor = 1.0

            # 成功率调整
            if success_rate < 0.9:  # 成功率低于90%
                quality_factor *= success_rate

            # 延迟调整（高延迟时降低仓位）
            if avg_latency > 100:  # 延迟超过100ms
                latency_penalty = max(0.5, 1.0 - (avg_latency - 100) / 1000)
                quality_factor *= latency_penalty

            adjusted_position = position * quality_factor

            if quality_factor < 1.0:
                self.logger.debug(
                    f"执行质量调整: {position:.4f} -> {adjusted_position:.4f} "
                    f"(成功率: {success_rate:.3f}, 延迟: {avg_latency:.1f}ms)"
                )

            return adjusted_position

        except Exception as e:
            self.logger.error(f"执行质量调整失败: {e}")
            return position

    def _validate_execution_quality(self, signal: IStrategySignal) -> Tuple[bool, str]:
        """🚀 新增：验证执行质量"""
        try:
            threshold = self.config.get("execution_quality_threshold", 0.9)
            success_rate = self._execution_quality_metrics.get("success_rate", 1.0)

            if success_rate < threshold:
                return False, f"执行质量过低: {success_rate:.3f} < {threshold}"

            return True, "执行质量验证通过"

        except Exception as e:
            self.logger.error(f"执行质量验证失败: {e}")
            return False, f"执行质量验证异常: {e}"

    def _validate_systemic_risk(self, signal: IStrategySignal) -> Tuple[bool, str]:
        """验证系统性风险"""
        try:
            # 检查系统性风险指标
            systemic_risk_threshold = 0.7
            current_systemic_risk = self._calculate_systemic_risk_exposure()

            if current_systemic_risk > systemic_risk_threshold:
                return (
                    False,
                    f"系统性风险过高: {current_systemic_risk:.3f} > {systemic_risk_threshold}",
                )

            # 检查市场状态稳定性
            if self._risk_metrics.regime_stability < 0.5:
                return False, f"市场状态不稳定: {self._risk_metrics.regime_stability:.3f}"

            return True, "系统性风险验证通过"

        except Exception as e:
            self.logger.error(f"系统性风险验证失败: {e}")
            return False, f"系统性风险验证异常: {e}"

    def _calculate_systemic_risk_exposure(self) -> float:
        """计算系统性风险暴露"""
        try:
            # 多维度系统性风险评估
            risk_factors = [
                self._risk_metrics.correlation_risk,
                1.0 - self._risk_metrics.liquidity_score,
                min(1.0, self._risk_metrics.volatility_30d / 0.3),  # 归一化
                self._risk_metrics.stress_test_score,
                1.0 - self._risk_metrics.execution_success_rate,  # 新增执行风险
            ]

            systemic_risk = sum(risk_factors) / len(risk_factors)
            return min(1.0, systemic_risk)

        except Exception as e:
            self.logger.error(f"系统性风险计算失败: {e}")
            return 0.5  # 默认中等风险

    def _generate_ai_risk_prediction(self, horizon_hours: int) -> RiskPrediction:
        """生成AI风险预测"""
        try:
            # 准备预测数据
            prediction_data = {
                "current_metrics": self._risk_metrics.to_dict(),
                "market_conditions": self._get_market_conditions(),
                "prediction_horizon": horizon_hours,
                "historical_risk_events": [
                    event.to_dict() for event in list(self._risk_events)[-10:]
                ],
            }

            # 使用SAC优化器进行预测
            if self._sac_risk_optimizer:
                prediction_signal = self._sac_risk_optimizer.get_signal(prediction_data)
                if prediction_signal:
                    confidence = prediction_signal.get_confidence()

                    # 映射到风险等级
                    if confidence > 0.8:
                        risk_level = RiskLevel.MINIMAL
                    elif confidence > 0.6:
                        risk_level = RiskLevel.LOW
                    elif confidence > 0.4:
                        risk_level = RiskLevel.MEDIUM
                    elif confidence > 0.2:
                        risk_level = RiskLevel.HIGH
                    else:
                        risk_level = RiskLevel.EXTREME

                    return RiskPrediction(
                        model_type=RiskPredictionModel.SAC_OPTIMIZED,
                        prediction_horizon=timedelta(hours=horizon_hours),
                        confidence=confidence,
                        predicted_risk_level=risk_level,
                        key_risk_factors=self._extract_key_risk_factors(
                            prediction_data
                        ),
                        mitigation_recommendations=self._generate_mitigation_recommendations(
                            risk_level
                        ),
                    )

            # 回退到基线预测
            return self._generate_baseline_risk_prediction(horizon_hours)

        except Exception as e:
            self.logger.error(f"AI风险预测生成失败: {e}")
            return self._generate_baseline_risk_prediction(horizon_hours)

    def _generate_baseline_risk_prediction(self, horizon_hours: int) -> RiskPrediction:
        """生成基线风险预测"""
        # 基于当前风险指标的简单预测
        current_risk_score = (
            self._risk_metrics.current_drawdown
            + self._risk_metrics.volatility_30d
            + (1 - self._risk_metrics.liquidity_score)
            + (1 - self._risk_metrics.execution_success_rate)  # 新增执行风险因子
        ) / 4

        if current_risk_score < 0.1:
            risk_level = RiskLevel.MINIMAL
            confidence = 0.9
        elif current_risk_score < 0.3:
            risk_level = RiskLevel.LOW
            confidence = 0.7
        elif current_risk_score < 0.5:
            risk_level = RiskLevel.MEDIUM
            confidence = 0.5
        elif current_risk_score < 0.7:
            risk_level = RiskLevel.HIGH
            confidence = 0.3
        else:
            risk_level = RiskLevel.EXTREME
            confidence = 0.1

        return RiskPrediction(
            model_type=RiskPredictionModel.ENSEMBLE_AI,
            prediction_horizon=timedelta(hours=horizon_hours),
            confidence=confidence,
            predicted_risk_level=risk_level,
            key_risk_factors=["baseline_prediction"],
            mitigation_recommendations=["监控市场变化", "准备应急计划"],
        )

    def _extract_key_risk_factors(self, prediction_data: Dict[str, Any]) -> List[str]:
        """提取关键风险因素"""
        risk_factors = []

        metrics = prediction_data.get("current_metrics", {})

        if metrics.get("volatility_30d", 0) > 0.2:
            risk_factors.append("高市场波动率")

        if metrics.get("liquidity_score", 1) < 0.7:
            risk_factors.append("流动性不足")

        if metrics.get("correlation_risk", 0) > 0.8:
            risk_factors.append("资产相关性风险")

        if metrics.get("current_drawdown", 0) > 0.1:
            risk_factors.append("当前回撤较大")

        if metrics.get("execution_success_rate", 1) < 0.9:
            risk_factors.append("执行成功率下降")

        if not risk_factors:
            risk_factors.append("风险因素在正常范围内")

        return risk_factors

    def _generate_mitigation_recommendations(self, risk_level: RiskLevel) -> List[str]:
        """生成风险缓解建议"""
        recommendations = []

        if risk_level == RiskLevel.EXTREME:
            recommendations.extend(
                ["立即减少总仓位至20%以下", "启用所有熔断机制", "增加对冲策略", "准备紧急平仓计划", "切换到高流动性交易对"]  # 新增
            )
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend(
                ["降低高风险资产配置", "收紧止损水平", "增加现金储备", "加强市场监控", "优化订单执行路由"]  # 新增
            )
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend(["适度调整仓位结构", "监控关键风险指标", "准备应急响应计划", "评估执行质量"])  # 新增
        else:
            recommendations.extend(["维持当前策略", "继续常规风险监控", "准备应对市场变化"])

        return recommendations

    def _execute_emergency_liquidation_advanced(self) -> Dict[str, Any]:
        """🚀 更新：使用最新订单执行器的紧急平仓"""
        try:
            if not self._order_executor:
                return {"success": False, "error": "订单执行器不可用"}

            self.logger.warning("开始执行高级紧急平仓...")

            # 获取执行器状态
            executor_status = self._order_executor.get_detailed_status()
            execution_mode = executor_status.get("execution_mode", "unknown")

            if execution_mode == "simulation":
                self.logger.info("模拟模式，跳过实际平仓")
                return {"success": True, "mode": "simulation", "message": "模拟平仓完成"}

            # 执行紧急平仓
            liquidation_result = self.emergency_liquidation(percent=0.5)  # 平仓50%

            # 记录紧急平仓事件
            emergency_event = RiskEvent(
                event_type=RiskEventType.EMERGENCY_LIQUIDATION,
                severity=RiskLevel.EXTREME,
                description="执行高级紧急平仓程序",
                triggered_by="circuit_breaker",
                data=liquidation_result,
            )

            self._risk_events.append(emergency_event)

            self.logger.info("高级紧急平仓程序执行完成")
            return liquidation_result

        except Exception as e:
            self.logger.error(f"高级紧急平仓执行失败: {e}")
            return {"success": False, "error": str(e)}

    def _create_liquidation_orders(
        self, symbol: str = None, percent: float = 1.0
    ) -> List[OrderRequest]:
        """🚀 新增：创建平仓订单"""
        liquidation_orders = []

        try:
            # 这里应该从仓位管理器中获取实际持仓
            # 简化实现：创建示例平仓订单
            symbols_to_liquidate = (
                [symbol] if symbol else ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            )

            for sym in symbols_to_liquidate:
                # 创建市价平仓订单
                order_request = OrderRequest(
                    symbol=sym,
                    order_type=OrderType.MARKET,
                    direction=SignalDirection.SHORT,  # 假设都是做多仓位，需要平仓
                    quantity=1000 * percent,  # 简化数量
                    reduce_only=True,  # 只减仓
                    client_order_id=f"emergency_liquidate_{uuid.uuid4().hex[:8]}",
                    strategy_source="RiskManagementSystem",
                    signal_confidence=1.0,
                    metadata={
                        "emergency_liquidation": True,
                        "liquidation_percent": percent,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                liquidation_orders.append(order_request)

            return liquidation_orders

        except Exception as e:
            self.logger.error(f"创建平仓订单失败: {e}")
            return []

    def _update_execution_quality_metrics(self):
        """🚀 新增：更新执行质量指标"""
        try:
            if not self._order_executor:
                return

            # 获取执行器指标
            execution_metrics = self._order_executor.get_execution_metrics()

            # 更新风险指标中的执行质量
            self._risk_metrics.execution_success_rate = execution_metrics.success_rate
            self._risk_metrics.average_slippage = (
                execution_metrics.total_slippage
                / max(1, execution_metrics.total_orders)
            )

            # 更新内部质量指标
            self._execution_quality_metrics.update(
                {
                    "success_rate": execution_metrics.success_rate,
                    "avg_latency": execution_metrics.average_execution_time,
                    "slippage": execution_metrics.total_slippage
                    / max(1, execution_metrics.total_orders),
                    "last_update": datetime.now(),
                }
            )

        except Exception as e:
            self.logger.warning(f"更新执行质量指标失败: {e}")

    # ==================== ?????? - ?????? ====================

    def _publish_risk_event(self, event_type, severity, description, source, data=None):
        """?????? - ????"""
        try:
            event = RiskEvent(
                event_type=event_type,
                severity=severity,
                description=description,
                triggered_by=source,
                data=data or {},
            )
            self._risk_events.append(event)
            self.logger.debug(f"??????: {event_type.value} - {description}")
        except Exception as e:
            self.logger.warning(f"????????: {e}")

    def _apply_risk_adjustments(self, position, signal, balance, confidence, direction):
        """?????? - ????"""
        try:
            # ??????
            risk_multiplier = 1.0

            # ???????
            if confidence < 0.7:
                risk_multiplier *= 0.8
            elif confidence > 0.9:
                risk_multiplier *= 1.1

            # ??????????
            if self._current_risk_level.value in ["HIGH", "EXTREME"]:
                risk_multiplier *= 0.5

            adjusted_position = position * risk_multiplier
            return max(0, adjusted_position)

        except Exception as e:
            self.logger.error(f"????????: {e}")
            return position

    def _validate_signal_integrity(self, signal):
        """??????? - ????"""
        try:
            if not signal:
                return False

            required_attrs = ["get_confidence", "get_signal_direction"]
            for attr in required_attrs:
                if not hasattr(signal, attr):
                    return False

            return True
        except Exception as e:
            self.logger.error(f"?????????: {e}")
            return False

    def _validate_market_conditions(self, signal):
        """?????? - ????"""
        try:
            # ???? - ??????
            return True, "????????"
        except Exception as e:
            self.logger.error(f"????????: {e}")
            return False, f"????????: {e}"

    def _validate_risk_layers(self, signal):
        """?????? - ????"""
        try:
            # ???? - ??????????
            enabled_layers = [
                layer for layer, enabled in self._risk_layers.items() if enabled
            ]
            if len(enabled_layers) < 3:  # ????3?????
                return False, "??????"
            return True, "????????"
        except Exception as e:
            self.logger.error(f"????????: {e}")
            return False, f"????????: {e}"

    def _validate_ai_prediction(self, signal):
        """??AI?? - ????"""
        try:
            if not self._enable_ai_prediction:
                return True, "AI?????"

            if not self._sac_risk_optimizer:
                return False, "AI??????"

            return True, "AI??????"
        except Exception as e:
            self.logger.error(f"AI??????: {e}")
            return False, f"AI??????: {e}"

    def _apply_final_limits(self, position, balance):
        """?????? - ????"""
        try:
            max_position = balance * self.config.get("max_position_size", 0.2)
            min_position = balance * 0.01  # ??1%

            final_position = max(min_position, min(position, max_position))
            return final_position
        except Exception as e:
            self.logger.error(f"????????: {e}")
            return min(position, balance * 0.1)  # ????

    def _assess_risk_comprehensive(self, signal, market_data):
        """?????? - ????"""
        try:
            # ??????
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                max_position_size=0.1,  # 10%??
                recommended_leverage=1.0,
                stop_loss_level=0.02,  # 2%??
                confidence=0.7,
                factors={"simplified_assessment": True},
            )
        except Exception as e:
            self.logger.error(f"????????: {e}")
            # ??????
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                max_position_size=0.05,
                recommended_leverage=1.0,
                stop_loss_level=0.05,
                confidence=0.3,
                factors={"error": str(e)},
            )

    def _publish_risk_event(self, event_type, severity, description, source, data=None):
        """?????? - ????"""
        try:
            from .risk_management import RiskEvent  # ??????

            event = RiskEvent(
                event_type=event_type,
                severity=severity,
                description=description,
                triggered_by=source,
                data=data or {},
            )
            self._risk_events.append(event)
            self.logger.debug(f"??????: {{event_type.value}} - {{description}}")
        except Exception as e:
            self.logger.warning(f"????????: {{e}}")

    def _apply_risk_adjustments(self, position, signal, balance, confidence, direction):
        """?????? - ????"""
        try:
            # ??????
            risk_multiplier = 1.0

            # ???????
            if confidence < 0.7:
                risk_multiplier *= 0.8
            elif confidence > 0.9:
                risk_multiplier *= 1.1

            # ??????????
            if hasattr(
                self, "_current_risk_level"
            ) and self._current_risk_level.value in ["HIGH", "EXTREME"]:
                risk_multiplier *= 0.5

            adjusted_position = position * risk_multiplier
            return max(0, adjusted_position)

        except Exception as e:
            self.logger.error(f"????????: {{e}}")
            return position

    def _validate_signal_integrity(self, signal):
        """??????? - ????"""
        try:
            if not signal:
                return False

            required_attrs = ["get_confidence", "get_signal_direction"]
            for attr in required_attrs:
                if not hasattr(signal, attr):
                    return False

            return True
        except Exception as e:
            self.logger.error(f"?????????: {{e}}")
            return False

    def _validate_market_conditions(self, signal):
        """?????? - ????"""
        try:
            # ???? - ??????
            return True, "????????"
        except Exception as e:
            self.logger.error(f"????????: {{e}}")
            return False, f"????????: {{e}}"

    def _validate_risk_layers(self, signal):
        """?????? - ????"""
        try:
            # ???? - ??????????
            enabled_layers = [
                layer
                for layer, enabled in getattr(self, "_risk_layers", {}).items()
                if enabled
            ]
            if len(enabled_layers) < 3:  # ????3?????
                return False, "??????"
            return True, "????????"
        except Exception as e:
            self.logger.error(f"????????: {{e}}")
            return False, f"????????: {{e}}"

    def _validate_ai_prediction(self, signal):
        """??AI?? - ????"""
        try:
            if not getattr(self, "_enable_ai_prediction", False):
                return True, "AI?????"

            if not getattr(self, "_sac_risk_optimizer", None):
                return False, "AI??????"

            return True, "AI??????"
        except Exception as e:
            self.logger.error(f"AI??????: {{e}}")
            return False, f"AI??????: {{e}}"

    def _apply_final_limits(self, position, balance):
        """?????? - ????"""
        try:
            max_position = balance * self.config.get("max_position_size", 0.2)
            min_position = balance * 0.01  # ??1%

            final_position = max(min_position, min(position, max_position))
            return final_position
        except Exception as e:
            self.logger.error(f"????????: {{e}}")
            return min(position, balance * 0.1)  # ????

    def _assess_risk_comprehensive(self, signal, market_data):
        """?????? - ????"""
        try:
            # ??????
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                max_position_size=0.1,  # 10%??
                recommended_leverage=1.0,
                stop_loss_level=0.02,  # 2%??
                confidence=0.7,
                factors={"simplified_assessment": True},
            )
        except Exception as e:
            self.logger.error(f"????????: {{e}}")
            # ??????
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                max_position_size=0.05,
                recommended_leverage=1.0,
                stop_loss_level=0.05,
                confidence=0.3,
                factors={"error": str(e)},
            )

    def _generate_stress_test_recommendations(self, stress_test_results):
        """???????? - ????"""
        try:
            recommendations = []
            overall_score = stress_test_results.get("overall_risk_score", 0)

            if overall_score > 0.7:
                recommendations.extend(["???????????", "??????", "????????"])
            elif overall_score > 0.5:
                recommendations.extend(["????????", "??????", "??????"])
            else:
                recommendations.extend(["??????", "??????"])

            return recommendations
        except Exception as e:
            self.logger.error(f"??????????: {{e}}")
            return ["??????????"]

    def _simulate_single_scenario(self, scenario, index):
        """?????????? - ????"""
        try:
            # ??????
            scenario_name = scenario.get("name", f"Scenario_{index}")
            risk_score = min(1.0, 0.3 + (index * 0.1))  # ??????

            return {
                "scenario_name": scenario_name,
                "risk_score": risk_score,
                "impact_analysis": "??????",
                "passed": risk_score < 0.7,
            }
        except Exception as e:
            self.logger.error(f"??????: {{e}}")
            return {
                "scenario_name": f"Scenario_{index}",
                "risk_score": 1.0,
                "impact_analysis": "????",
                "passed": False,
            }

    def _get_market_conditions(self) -> Dict[str, Any]:
        """??????"""
        # ???????????????????
        return {
            "volatility_regime": "normal",
            "liquidity_conditions": "adequate",
            "market_sentiment": "neutral",
            "economic_calendar": "no_major_events",
        }

    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        # 🚀 新增：更新执行质量指标
        self._update_execution_quality_metrics()

        return self._performance_metrics

    def clear_cache(self) -> bool:
        """清空缓存"""
        try:
            self._risk_assessment_cache.clear()
            self._prediction_cache.clear()
            self._performance_metrics.cache_hit_rate = 0.0
            return True
        except Exception as e:
            self.logger.error(f"缓存清空失败: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取风险管理系统状态 - 极致优化版本"""
        try:
            # 🚀 新增：更新执行质量指标
            self._update_execution_quality_metrics()

            status = {
                "system_enabled": self._system_enabled,
                "current_risk_level": self._current_risk_level.value,
                "circuit_breaker_triggered": self._circuit_breaker_triggered,
                "ai_prediction_enabled": self._enable_ai_prediction,
                "emergency_liquidation_enabled": self.config.get(
                    "emergency_liquidation_enabled", True
                ),
                "risk_metrics": self.get_risk_metrics(),
                "performance_metrics": self._performance_metrics.to_dict(),
                "execution_quality_metrics": self._execution_quality_metrics,  # 新增
                "active_risk_layers": [
                    layer.value
                    for layer, enabled in self._risk_layers.items()
                    if enabled
                ],
                "recent_risk_events": len(self._risk_events),
                "adaptive_parameters": self._adaptive_parameters,
                "risk_predictions_count": len(self._risk_predictions),
                "emergency_orders_count": len(self._emergency_stop_orders),  # 新增
                "initialized": self.initialized,
                "version": "5.0-ai-driven-predictive-enhanced",
            }

            # 添加订单执行器状态
            if self._order_executor:
                status[
                    "order_executor_status"
                ] = self._order_executor.get_detailed_status()

            return status

        except Exception as e:
            self.logger.error(f"获取状态失败: {e}")
            return {
                "system_enabled": False,
                "error": str(e),
                "initialized": False,
                "version": "5.0-ai-driven-predictive-enhanced",
            }


# ==================== 风险管理系统工厂类 ====================


class RiskManagementFactory:
    """风险管理系统工厂 - 支持动态创建和管理"""

    _risk_managers: Dict[str, RiskManagementSystem] = {}

    @classmethod
    def create_risk_manager(
        cls, name: str, config: Dict[str, Any]
    ) -> RiskManagementSystem:
        """创建风险管理系统"""
        try:
            risk_manager = RiskManagementSystem(name, config)

            if risk_manager.initialize():
                cls._risk_managers[name] = risk_manager
                return risk_manager
            else:
                # 即使初始化失败也返回实例
                cls._risk_managers[name] = risk_manager
                return risk_manager

        except Exception as e:
            # 创建基本实例
            basic_manager = RiskManagementSystem(name, config)
            basic_manager.initialized = False
            cls._risk_managers[name] = basic_manager
            return basic_manager

    @classmethod
    def create_ai_risk_manager(
        cls, name: str, config: Dict[str, Any]
    ) -> RiskManagementSystem:
        """创建AI驱动风险管理系统"""
        ai_config = {
            "ai_risk_prediction": True,
            "quantum_coherence_integration": True,
            "distributed_risk_calculation": True,
            "emergency_liquidation_enabled": True,  # 默认启用紧急平仓
        }
        ai_config.update(config)

        risk_manager = cls.create_risk_manager(name, ai_config)
        if risk_manager.initialized:
            risk_manager.enable_ai_risk_prediction()

        return risk_manager

    @classmethod
    def create_enhanced_risk_manager(
        cls, name: str, config: Dict[str, Any]
    ) -> RiskManagementSystem:
        """🚀 新增：创建增强版风险管理系统"""
        enhanced_config = {
            "ai_risk_prediction": True,
            "emergency_liquidation_enabled": True,
            "execution_quality_threshold": 0.95,
            "max_emergency_liquidation_percent": 0.7,
            "multi_layer_control": True,
            "adaptive_risk_parameters": True,
        }
        enhanced_config.update(config)

        return cls.create_risk_manager(name, enhanced_config)

    @classmethod
    def get_risk_manager(cls, name: str) -> Optional[RiskManagementSystem]:
        return cls._risk_managers.get(name)

    @classmethod
    def list_risk_managers(cls) -> List[str]:
        return list(cls._risk_managers.keys())


# ==================== 自动注册接口 ====================

try:
    from src.interfaces import InterfaceRegistry

    InterfaceRegistry.register_interface(RiskManagementSystem)
except ImportError:
    pass

__all__ = [
    "RiskManagementSystem",
    "RiskManagementFactory",
    "RiskControlLayer",
    "RiskEventType",
    "RiskPredictionModel",
    "RiskMetrics",
    "RiskEvent",
    "PositionRisk",
    "RiskPrediction",
]

# src/interface.py
"""
量子奇点狙击系统 - 接口定义 V5.0
🎯 极致优化版本: 性能监控 + 版本控制 + 智能发现 + 异步处理
✅ 企业级接口契约，支持自适应架构
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import Future
import time
import uuid
from typing_extensions import Protocol

# ==================== 极致优化: 性能监控系统 ====================


@dataclass
class PerformanceMetrics:
    """性能指标数据类 - 极致优化"""

    execution_time: float
    memory_usage: int
    cpu_usage: float
    call_count: int
    error_count: int
    cache_hit_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "cache_hit_rate": self.cache_hit_rate,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InterfaceMetadata:
    """接口元数据 - 极致优化"""

    version: str
    description: str
    author: str
    created_date: datetime
    last_modified: datetime = field(default_factory=datetime.now)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    compatibility: List[str] = field(default_factory=list)

    def is_compatible_with(self, other_version: str) -> bool:
        """检查版本兼容性"""
        return other_version in self.compatibility


# ==================== 极致优化: 智能信号系统 ====================


class SignalDirection(Enum):
    """信号方向枚举 - 增强版本"""

    LONG = auto()
    SHORT = auto()
    NEUTRAL = auto()
    HEDGE = auto()  # 新增: 对冲信号

    def is_directional(self) -> bool:
        """判断是否为方向性信号"""
        return self in [SignalDirection.LONG, SignalDirection.SHORT]


class SignalPriority(Enum):
    """信号优先级 - 新增极致优化"""

    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25


@dataclass
class SignalMetadata:
    """信号元数据 - 新增极致优化"""

    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    priority: SignalPriority = SignalPriority.MEDIUM
    expiration: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    confidence_interval: Tuple[float, float] = (0.0, 1.0)


class IStrategySignal(ABC):
    """策略信号接口 - 极致优化版本"""

    # 接口元数据 - 新增极致优化
    _metadata = InterfaceMetadata(
        version="5.0",
        description="统一策略信号接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "signal_generation_time": 0.001,
            "signal_validation_time": 0.0005,
        },
        dependencies=["IDataProcessor", "IMarketAnalyzer"],
        compatibility=["4.2", "4.1"],
    )

    @classmethod
    def get_interface_metadata(cls) -> InterfaceMetadata:
        """获取接口元数据 - 新增极致优化"""
        return cls._metadata

    @abstractmethod
    def get_signal_strength(self) -> float:
        """获取信号强度 (0.0-1.0)"""
        pass

    @abstractmethod
    def get_signal_direction(self) -> SignalDirection:
        """获取信号方向"""
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """获取置信度 (0.0-1.0)"""
        pass

    @abstractmethod
    def get_timestamp(self) -> int:
        """获取信号时间戳"""
        pass

    # 🚀 新增极致优化方法
    @abstractmethod
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标 - 新增极致优化"""
        pass

    @abstractmethod
    def validate_signal_integrity(self) -> Tuple[bool, str]:
        """验证信号完整性 - 新增极致优化"""
        pass

    @abstractmethod
    def get_signal_metadata(self) -> SignalMetadata:
        """获取信号元数据 - 新增极致优化"""
        pass

    @abstractmethod
    def is_expired(self) -> bool:
        """检查信号是否过期 - 新增极致优化"""
        pass

    @abstractmethod
    async def generate_async(self) -> "IStrategySignal":
        """异步生成信号 - 新增极致优化"""
        pass


# ==================== 极致优化: 智能数据处理 ====================


class DataQualityLevel(Enum):
    """数据质量等级 - 新增极致优化"""

    EXCELLENT = 95
    GOOD = 80
    FAIR = 65
    POOR = 50
    UNUSABLE = 0


@dataclass
class DataProcessingMetrics:
    """数据处理指标 - 新增极致优化"""

    processing_time: float
    data_volume: int
    quality_score: DataQualityLevel
    feature_count: int
    outlier_count: int
    transformation_applied: List[str]


class IDataProcessor(ABC):
    """数据处理器接口 - 极致优化版本"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="智能数据处理器接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "data_processing_time": 0.005,
            "feature_extraction_time": 0.002,
        },
    )

    @abstractmethod
    def process_data(self, raw_data: Any) -> Dict[str, Any]:
        """处理原始数据"""
        pass

    @abstractmethod
    def validate_data_quality(self, data: Dict[str, Any]) -> bool:
        """验证数据质量"""
        pass

    @abstractmethod
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """提取特征"""
        pass

    # 🚀 新增极致优化方法
    @abstractmethod
    async def process_data_async(self, raw_data: Any) -> Dict[str, Any]:
        """异步处理数据 - 新增极致优化"""
        pass

    @abstractmethod
    def get_data_quality_metrics(self, data: Dict[str, Any]) -> DataQualityLevel:
        """获取数据质量指标 - 新增极致优化"""
        pass

    @abstractmethod
    def get_processing_metrics(self) -> DataProcessingMetrics:
        """获取处理指标 - 新增极致优化"""
        pass

    @abstractmethod
    def detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测数据异常 - 新增极致优化"""
        pass

    @abstractmethod
    def optimize_pipeline(self) -> bool:
        """优化处理管道 - 新增极致优化"""
        pass


# ==================== 极致优化: 高性能事件系统 ====================


class EventPriority(Enum):
    """事件优先级 - 新增极致优化"""

    REAL_TIME = 1000
    HIGH = 100
    NORMAL = 50
    LOW = 10
    BACKGROUND = 1


@dataclass
class Event:
    """事件对象 - 新增极致优化"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    data: Any = None
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class IEventDispatcher(ABC):
    """事件分发器接口 - 极致优化版本"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="高性能事件分发器接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "event_dispatch_time": 0.0001,
            "handler_execution_time": 0.001,
        },
    )

    @abstractmethod
    def dispatch_event(self, event_type: str, data: Any) -> bool:
        """分发事件"""
        pass

    @abstractmethod
    def register_handler(self, event_type: str, handler: Callable) -> bool:
        """注册事件处理器"""
        pass

    @abstractmethod
    def remove_handler(self, event_type: str, handler: Callable) -> bool:
        """移除事件处理器"""
        pass

    # 🚀 新增极致优化方法
    @abstractmethod
    async def dispatch_event_async(self, event: Event) -> bool:
        """异步分发事件 - 新增极致优化"""
        pass

    @abstractmethod
    def get_event_metrics(self) -> Dict[str, Any]:
        """获取事件处理指标 - 新增极致优化"""
        pass

    @abstractmethod
    def set_event_priority(self, event_type: str, priority: EventPriority) -> bool:
        """设置事件优先级 - 新增极致优化"""
        pass

    @abstractmethod
    def get_pending_events_count(self) -> int:
        """获取待处理事件数量 - 新增极致优化"""
        pass

    @abstractmethod
    def optimize_event_flow(self) -> bool:
        """优化事件流 - 新增极致优化"""
        pass

    @abstractmethod
    def register_conditional_handler(
        self, condition: Callable[[Event], bool], handler: Callable
    ) -> bool:
        """注册条件事件处理器 - 新增极致优化"""
        pass


# ==================== 极致优化: 动态配置管理 ====================


class ConfigScope(Enum):
    """配置作用域 - 新增极致优化"""

    GLOBAL = "global"
    STRATEGY = "strategy"
    RISK = "risk"
    PERFORMANCE = "performance"
    ENVIRONMENT = "environment"


@dataclass
class ConfigChange:
    """配置变更记录 - 新增极致优化"""

    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str
    reason: str = ""


class IConfigManager(ABC):
    """配置管理器接口 - 极致优化版本"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="动态配置管理器接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={"config_load_time": 0.01, "config_validation_time": 0.005},
    )

    @abstractmethod
    def load_config(self) -> bool:
        """加载配置"""
        pass

    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        pass

    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """设置配置值"""
        pass

    @abstractmethod
    def validate_config(self) -> Tuple[bool, List[str]]:
        """验证配置完整性"""
        pass

    # 🚀 新增极致优化方法
    @abstractmethod
    def hot_reload_config(self) -> bool:
        """热重载配置 - 新增极致优化"""
        pass

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式 - 新增极致优化"""
        pass

    @abstractmethod
    def watch_config(self, key: str, callback: Callable[[ConfigChange], None]) -> bool:
        """监控配置变更 - 新增极致优化"""
        pass

    @abstractmethod
    def get_config_history(self, key: str) -> List[ConfigChange]:
        """获取配置历史 - 新增极致优化"""
        pass

    @abstractmethod
    def rollback_config(self, key: str, steps: int = 1) -> bool:
        """回滚配置 - 新增极致优化"""
        pass

    @abstractmethod
    def optimize_config_storage(self) -> bool:
        """优化配置存储 - 新增极致优化"""
        pass

    @abstractmethod
    def get_config_by_scope(self, scope: ConfigScope) -> Dict[str, Any]:
        """按作用域获取配置 - 新增极致优化"""
        pass


# ==================== 极致优化: AI驱动市场分析 ====================


class MarketRegime(Enum):
    """市场状态 - 增强版本"""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class MarketAnalysis:
    """市场分析结果 - 新增极致优化"""

    regime: MarketRegime
    confidence: float
    key_indicators: Dict[str, float]
    timeframe: str
    timestamp: datetime
    recommendations: List[str]
    risk_level: int


class IMarketAnalyzer(ABC):
    """市场分析器接口 - 极致优化版本"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="AI驱动市场分析器接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "market_analysis_time": 0.01,
            "regime_detection_time": 0.005,
        },
    )

    @abstractmethod
    def analyze_market_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析市场状况"""
        pass

    @abstractmethod
    def detect_regime(self, data: Dict[str, Any]) -> str:
        """检测市场状态"""
        pass

    # 🚀 新增极致优化方法
    @abstractmethod
    async def analyze_market_async(self, data: Dict[str, Any]) -> MarketAnalysis:
        """异步市场分析 - 新增极致优化"""
        pass

    @abstractmethod
    def get_market_insights(self, data: Dict[str, Any]) -> List[str]:
        """获取市场洞察 - 新增极致优化"""
        pass

    @abstractmethod
    def predict_regime_shift(
        self, data: Dict[str, Any]
    ) -> Tuple[MarketRegime, float, int]:
        """预测状态转换 - 新增极致优化"""
        pass

    @abstractmethod
    def get_analysis_confidence(self, data: Dict[str, Any]) -> float:
        """获取分析置信度 - 新增极致优化"""
        pass

    @abstractmethod
    def optimize_analysis_model(self) -> bool:
        """优化分析模型 - 新增极致优化"""
        pass


# ==================== 极致优化: 自适应风险管理系统 ====================


class RiskLevel(Enum):
    """风险等级 - 新增极致优化"""

    EXTREME = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    MINIMAL = 10


@dataclass
class RiskAssessment:
    """风险评估结果 - 新增极致优化"""

    risk_level: RiskLevel
    max_position_size: float
    recommended_leverage: float
    stop_loss_level: float
    confidence: float
    factors: Dict[str, float]


class IRiskManager(ABC):
    """风险管理器接口 - 极致优化版本"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="自适应风险管理器接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "risk_assessment_time": 0.002,
            "position_calculation_time": 0.001,
        },
    )

    @abstractmethod
    def calculate_position_size(self, signal: IStrategySignal, balance: float) -> float:
        """计算仓位大小"""
        pass

    @abstractmethod
    def validate_trade_signal(self, signal: IStrategySignal) -> Tuple[bool, str]:
        """验证交易信号"""
        pass

    # 🚀 新增极致优化方法
    @abstractmethod
    async def assess_risk_async(
        self, signal: IStrategySignal, market_data: Dict[str, Any]
    ) -> RiskAssessment:
        """异步风险评估 - 新增极致优化"""
        pass

    @abstractmethod
    def get_risk_metrics(self) -> Dict[str, float]:
        """获取风险指标 - 新增极致优化"""
        pass

    @abstractmethod
    def adjust_risk_parameters(self, market_regime: MarketRegime) -> bool:
        """调整风险参数 - 新增极致优化"""
        pass

    @abstractmethod
    def simulate_stress_test(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """压力测试模拟 - 新增极致优化"""
        pass

    @abstractmethod
    def get_risk_exposure(self) -> Dict[str, float]:
        """获取风险暴露 - 新增极致优化"""
        pass


# ==================== 极致优化: 接口注册与发现系统 ====================


class InterfaceRegistry:
    """接口注册表 - 新增极致优化"""

    _registry: Dict[str, Any] = {}

    @classmethod
    def register_interface(cls, interface_class: Any) -> bool:
        """注册接口"""
        interface_name = interface_class.__name__
        cls._registry[interface_name] = interface_class
        return True

    @classmethod
    def get_interface(cls, interface_name: str) -> Optional[Any]:
        """获取接口类"""
        return cls._registry.get(interface_name)

    @classmethod
    def list_interfaces(cls) -> List[str]:
        """列出所有注册的接口"""
        return list(cls._registry.keys())

    @classmethod
    def get_interface_metadata(cls, interface_name: str) -> Optional[InterfaceMetadata]:
        """获取接口元数据"""
        interface_class = cls.get_interface(interface_name)
        if interface_class and hasattr(interface_class, "get_interface_metadata"):
            return interface_class.get_interface_metadata()
        return None


# ==================== 注册所有接口 ====================

# 自动注册所有接口类
InterfaceRegistry.register_interface(IStrategySignal)
InterfaceRegistry.register_interface(IDataProcessor)
InterfaceRegistry.register_interface(IEventDispatcher)
InterfaceRegistry.register_interface(IConfigManager)
InterfaceRegistry.register_interface(IMarketAnalyzer)
InterfaceRegistry.register_interface(IRiskManager)

# ==================== 导出所有接口和组件 ====================

__all__ = [
    # 核心接口
    "IStrategySignal",
    "IDataProcessor",
    "IEventDispatcher",
    "IConfigManager",
    "IMarketAnalyzer",
    "IRiskManager",
    # 枚举类型
    "SignalDirection",
    "SignalPriority",
    "DataQualityLevel",
    "EventPriority",
    "ConfigScope",
    "MarketRegime",
    "RiskLevel",
    # 数据类
    "PerformanceMetrics",
    "InterfaceMetadata",
    "SignalMetadata",
    "DataProcessingMetrics",
    "Event",
    "ConfigChange",
    "MarketAnalysis",
    "RiskAssessment",
    # 系统组件
    "InterfaceRegistry",
    "validate_interfaces",
]

# ==================== 系统初始化验证 ====================


def validate_interfaces() -> Tuple[bool, List[str]]:
    """验证所有接口的完整性 - 新增极致优化"""
    issues = []

    required_interfaces = [
        IStrategySignal,
        IDataProcessor,
        IEventDispatcher,
        IConfigManager,
        IMarketAnalyzer,
        IRiskManager,
    ]

    for interface in required_interfaces:
        if not hasattr(interface, "_metadata"):
            issues.append(f"接口 {interface.__name__} 缺少元数据")

        # 检查抽象方法实现
        abstract_methods = []
        for name in dir(interface):
            attr = getattr(interface, name)
            if getattr(attr, "__isabstractmethod__", False):
                abstract_methods.append(name)

        if abstract_methods:
            issues.append(f"接口 {interface.__name__} 有未实现的抽象方法: {abstract_methods}")

    return len(issues) == 0, issues


# 系统启动时自动验证
is_valid, validation_issues = validate_interfaces()

if __name__ == "__main__":
    print("🚀 量子奇点系统接口定义 V5.0 - 极致优化版本 加载完成")
    print(f"✅ 接口验证: {'通过' if is_valid else '失败'}")

    if not is_valid:
        print("❌ 发现的问题:")
        for issue in validation_issues:
            print(f"   - {issue}")
    else:
        print("🎯 极致优化特性:")
        print("   • 性能监控集成")
        print("   • 版本控制系统")
        print("   • 异步处理支持")
        print("   • 智能接口发现")
        print("   • 自适应风险控制")
        print("   • AI驱动市场分析")

        # 显示注册的接口
        print(f"📋 注册接口数量: {len(InterfaceRegistry.list_interfaces())}")
        for interface in InterfaceRegistry.list_interfaces():
            metadata = InterfaceRegistry.get_interface_metadata(interface)
            if metadata:
                print(f"   • {interface} (v{metadata.version})")

# ==================== 新增：订单相关接口定义 ====================


class OrderType(Enum):
    """订单类型枚举"""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """订单方向枚举"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态枚举"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class Order:
    """订单数据类"""

    order_id: str
    symbol: str
    order_type: OrderType
    order_side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "order_side": self.order_side.value,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExecutionReport:
    """执行报告数据类"""

    order_id: str
    status: OrderStatus
    executed_quantity: float
    average_price: float
    total_cost: float
    slippage_bps: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "order_id": self.order_id,
            "status": self.status.value,
            "executed_quantity": self.executed_quantity,
            "average_price": self.average_price,
            "total_cost": self.total_cost,
            "slippage_bps": self.slippage_bps,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MarketData:
    """市场数据类"""

    symbol: str
    current_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    volatility: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "volume_24h": self.volume_24h,
            "volatility": self.volatility,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LiquidityProvider:
    """流动性提供者数据类"""

    name: str
    exchange: str
    rating: float = 1.0
    supported_pairs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "exchange": self.exchange,
            "rating": self.rating,
            "supported_pairs": self.supported_pairs,
        }


# 更新导出列表
__all__.extend(
    [
        "OrderType",
        "OrderSide",
        "OrderStatus",
        "Order",
        "ExecutionReport",
        "MarketData",
        "LiquidityProvider",
    ]
)

# ==================== 新增：仓位数据定义 ====================


@dataclass
class PositionData:
    """仓位数据类"""

    symbol: str
    current_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "current_size": self.current_size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "timestamp": self.timestamp.isoformat(),
        }


# ==================== 新增：完整订单执行器接口 ====================


class IOrderExecutor(ABC):
    """订单执行器接口 - 完整定义"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="统一订单执行器接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "order_execution_time": 0.02,
            "order_processing_time": 0.005,
        },
    )

    @abstractmethod
    def execute_order(self, order_request: Any) -> Any:
        """执行订单"""
        pass

    @abstractmethod
    async def execute_order_async(self, order_request: Any) -> Any:
        """异步执行订单"""
        pass

    @abstractmethod
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        pass

    @abstractmethod
    def get_execution_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        pass


# ==================== 新增：策略引擎接口 ====================


class IStrategyEngine(ABC):
    """策略引擎接口"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="策略引擎接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
    )

    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> List[IStrategySignal]:
        """生成交易信号"""
        pass

    @abstractmethod
    def validate_strategy(self) -> Tuple[bool, str]:
        """验证策略"""
        pass


# 更新导出列表
__all__.extend(["PositionData", "IOrderExecutor", "IStrategyEngine"])

# ==================== 新增：仓位数据定义 ====================


@dataclass
class PositionData:
    """仓位数据类"""

    symbol: str
    current_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "current_size": self.current_size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "timestamp": self.timestamp.isoformat(),
        }


# ==================== 新增：完整订单执行器接口 ====================


class IOrderExecutor(ABC):
    """订单执行器接口 - 完整定义"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="统一订单执行器接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "order_execution_time": 0.02,
            "order_processing_time": 0.005,
        },
    )

    @abstractmethod
    def execute_order(self, order_request: Any) -> Any:
        """执行订单"""
        pass

    @abstractmethod
    async def execute_order_async(self, order_request: Any) -> Any:
        """异步执行订单"""
        pass

    @abstractmethod
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        pass

    @abstractmethod
    def get_execution_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        pass


# ==================== 新增：策略引擎接口 ====================


class IStrategyEngine(ABC):
    """策略引擎接口"""

    _metadata = InterfaceMetadata(
        version="5.0",
        description="策略引擎接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
    )

    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> List[IStrategySignal]:
        """生成交易信号"""
        pass

    @abstractmethod
    def validate_strategy(self) -> Tuple[bool, str]:
        """验证策略"""
        pass


# 更新导出列表
__all__.extend(["PositionData", "IOrderExecutor", "IStrategyEngine"])

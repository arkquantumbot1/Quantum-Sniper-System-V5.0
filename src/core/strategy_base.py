# src/core/strategy_base.py
"""量子奇点系统 - 策略基类 V5.0 (完全重新开发 + 极致优化)"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import logging
import asyncio
from datetime import datetime
import uuid

# 导入极致优化的接口定义
from interfaces import (
    IStrategySignal, SignalDirection, SignalPriority, PerformanceMetrics,
    InterfaceMetadata, SignalMetadata, DataQualityLevel, MarketRegime
)

class StrategySignal(IStrategySignal):
    """策略信号实现类 - 极致优化版本"""
    
    def __init__(self, signal_type: str, confidence: float, data: Dict[str, Any],
                 direction: SignalDirection = SignalDirection.NEUTRAL,
                 metadata: Optional[SignalMetadata] = None):
        self._signal_type = signal_type
        self._confidence = max(0.0, min(1.0, confidence))  # 确保在0-1范围内
        self._data = data or {}
        self._direction = direction
        self._timestamp = int(datetime.now().timestamp() * 1000)  # 毫秒时间戳
        
        # 信号元数据 - 极致优化
        self._metadata = metadata or SignalMetadata()
        self._performance_metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0,
            cpu_usage=0.0,
            call_count=1,
            error_count=0,
            cache_hit_rate=0.0
        )
    
    def get_signal_strength(self) -> float:
        """获取信号强度 (0.0-1.0) - 极致优化"""
        base_strength = self._confidence
        # 基于信号类型和数据的增强计算
        if self._signal_type == "TREND_REVERSAL":
            base_strength *= 1.2  # 趋势反转信号增强
        elif self._signal_type == "BREAKOUT":
            base_strength *= 1.1  # 突破信号增强
        
        return min(1.0, base_strength)
    
    def get_signal_direction(self) -> SignalDirection:
        """获取信号方向 - 极致优化"""
        return self._direction
    
    def get_confidence(self) -> float:
        """获取置信度 (0.0-1.0) - 极致优化"""
        return self._confidence
    
    def get_timestamp(self) -> int:
        """获取信号时间戳 - 极致优化"""
        return self._timestamp
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标 - 新增极致优化"""
        return self._performance_metrics
    
    def validate_signal_integrity(self) -> Tuple[bool, str]:
        """验证信号完整性 - 新增极致优化"""
        if self._confidence <= 0:
            return False, "信号置信度必须大于0"
        
        if not self._signal_type:
            return False, "信号类型不能为空"
        
        if self._timestamp <= 0:
            return False, "无效的时间戳"
        
        # 检查数据质量
        required_fields = self._metadata.tags or []
        for field in required_fields:
            if field not in self._data:
                return False, f"缺少必需字段: {field}"
        
        return True, "信号完整性验证通过"
    
    def get_signal_metadata(self) -> SignalMetadata:
        """获取信号元数据 - 新增极致优化"""
        return self._metadata
    
    def is_expired(self) -> bool:
        """检查信号是否过期 - 新增极致优化"""
        if not self._metadata.expiration:
            return False
        
        current_time = datetime.now()
        return current_time >= self._metadata.expiration
    
    async def generate_async(self) -> 'StrategySignal':
        """异步生成信号 - 新增极致优化"""
        # 模拟异步处理
        await asyncio.sleep(0.001)  # 最小延迟
        return self
    
    def __str__(self) -> str:
        return (f"StrategySignal(type={self._signal_type}, "
                f"confidence={self._confidence:.3f}, "
                f"direction={self._direction.name})")

class BaseStrategy(ABC):
    """策略基类 V5.0 - 完全重新开发 + 极致优化"""
    
    # 接口元数据 - 新增极致优化
    _metadata = InterfaceMetadata(
        version="5.0",
        description="统一策略基类接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "signal_generation_time": 0.001,
            "strategy_initialization_time": 0.01
        },
        dependencies=["IDataProcessor", "IMarketAnalyzer", "IRiskManager"],
        compatibility=["4.2", "4.1"]
    )
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.initialized = False
        self.logger = logging.getLogger(f"strategy.{name}")
        
        # 性能监控 - 新增极致优化
        self._performance_metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0,
            cpu_usage=0.0,
            call_count=0,
            error_count=0,
            cache_hit_rate=0.0
        )
        
        # 策略状态 - 极致优化
        self._status = {
            "name": name,
            "initialized": False,
            "last_signal_time": None,
            "signal_count": 0,
            "error_count": 0,
            "performance_score": 0.0
        }
        
        # 智能缓存 - 新增极致优化
        self._signal_cache: Dict[str, IStrategySignal] = {}
        self._cache_ttl = 60  # 缓存TTL（秒）
    
    @classmethod
    def get_interface_metadata(cls) -> InterfaceMetadata:
        """获取接口元数据 - 新增极致优化"""
        return cls._metadata
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化策略 - 极致优化版本"""
        pass
    
    @abstractmethod
    def get_signal(self, data: Any) -> Optional[IStrategySignal]:
        """获取交易信号 - 极致优化版本"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取策略状态 - 极致优化版本"""
        pass
    
    def validate_parameters(self) -> bool:
        """验证参数 - 极致优化版本"""
        required_params = ["name", "enabled", "risk_level"]
        if not all(param in self.config for param in required_params):
            self.logger.error(f"策略 {self.name} 缺少必需参数: {required_params}")
            return False
        
        # 验证风险级别
        risk_level = self.config.get("risk_level", "medium")
        valid_risk_levels = ["low", "medium", "high", "extreme"]
        if risk_level not in valid_risk_levels:
            self.logger.error(f"无效的风险级别: {risk_level}")
            return False
        
        return True
    
    # 🚀 新增极致优化方法
    
    async def get_signal_async(self, data: Any) -> Optional[IStrategySignal]:
        """异步获取交易信号 - 新增极致优化"""
        try:
            # 使用缓存提高性能
            cache_key = self._generate_cache_key(data)
            if cache_key in self._signal_cache:
                cached_signal = self._signal_cache[cache_key]
                if not cached_signal.is_expired():
                    self._performance_metrics.cache_hit_rate += 1
                    return cached_signal
            
            # 生成新信号
            signal = await asyncio.get_event_loop().run_in_executor(
                None, self.get_signal, data
            )
            
            if signal and isinstance(signal, IStrategySignal):
                # 缓存信号
                self._signal_cache[cache_key] = signal
                self._performance_metrics.call_count += 1
                self._status["signal_count"] += 1
                self._status["last_signal_time"] = datetime.now().isoformat()
            
            return signal
            
        except Exception as e:
            self.logger.error(f"异步获取信号失败: {e}")
            self._performance_metrics.error_count += 1
            self._status["error_count"] += 1
            return None
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标 - 新增极致优化"""
        return self._performance_metrics
    
    def optimize_strategy(self) -> bool:
        """优化策略性能 - 新增极致优化"""
        try:
            # 清理过期缓存
            current_time = datetime.now()
            expired_keys = []
            for key, signal in self._signal_cache.items():
                if signal.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._signal_cache[key]
            
            # 更新性能分数
            total_calls = self._performance_metrics.call_count
            if total_calls > 0:
                error_rate = self._performance_metrics.error_count / total_calls
                cache_hit_rate = self._performance_metrics.cache_hit_rate / total_calls
                self._status["performance_score"] = (1 - error_rate) * cache_hit_rate
            
            self.logger.info(f"策略 {self.name} 优化完成，性能分数: {self._status['performance_score']:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"策略优化失败: {e}")
            return False
    
    def validate_strategy_integrity(self) -> Tuple[bool, List[str]]:
        """验证策略完整性 - 新增极致优化"""
        issues = []
        
        # 验证配置完整性
        if not self.validate_parameters():
            issues.append("参数验证失败")
        
        # 验证初始化状态
        if not self.initialized:
            issues.append("策略未初始化")
        
        # 验证性能指标
        if self._performance_metrics.error_count > 100:
            issues.append("错误计数过高，需要检查策略逻辑")
        
        return len(issues) == 0, issues
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """获取详细状态信息 - 新增极致优化"""
        base_status = self.get_status()
        detailed_status = {
            **base_status,
            "performance_metrics": self._performance_metrics.to_dict(),
            "cache_size": len(self._signal_cache),
            "initialization_time": self.config.get("initialization_time"),
            "strategy_version": self._metadata.version,
            "compatibility": self._metadata.compatibility
        }
        return detailed_status
    
    def _generate_cache_key(self, data: Any) -> str:
        """生成缓存键 - 新增极致优化"""
        import hashlib
        data_str = str(data).encode('utf-8')
        return hashlib.md5(data_str).hexdigest()
    
    def __str__(self) -> str:
        return f"StrategyV5({self.name}, v{self._metadata.version})"
    
    def __repr__(self) -> str:
        return (f"BaseStrategy(name={self.name}, initialized={self.initialized}, "
                f"signal_count={self._status['signal_count']})")

class StrategyError(Exception):
    """策略异常基类 - 极致优化版本"""
    
    def __init__(self, message: str, strategy_name: str = None, error_code: str = None):
        self.message = message
        self.strategy_name = strategy_name
        self.error_code = error_code
        self.timestamp = datetime.now()
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        base_msg = f"策略错误: {self.message}"
        if self.strategy_name:
            base_msg += f" [策略: {self.strategy_name}]"
        if self.error_code:
            base_msg += f" [错误码: {self.error_code}]"
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式 - 新增极致优化"""
        return {
            "message": self.message,
            "strategy_name": self.strategy_name,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat()
        }

# 策略工厂类 - 新增极致优化
class StrategyFactory:
    """策略工厂 - 支持动态策略发现和创建"""
    
    _strategies: Dict[str, type] = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> bool:
        """注册策略类"""
        if not issubclass(strategy_class, BaseStrategy):
            raise StrategyError(f"注册的策略类必须继承自 BaseStrategy: {strategy_class}")
        
        cls._strategies[name] = strategy_class
        return True
    
    @classmethod
    def create_strategy(cls, name: str, config: Dict[str, Any]) -> BaseStrategy:
        """创建策略实例"""
        if name not in cls._strategies:
            raise StrategyError(f"未找到策略: {name}")
        
        strategy_class = cls._strategies[name]
        return strategy_class(name, config)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有注册的策略"""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_metadata(cls, name: str) -> Optional[InterfaceMetadata]:
        """获取策略元数据"""
        if name in cls._strategies:
            strategy_class = cls._strategies[name]
            if hasattr(strategy_class, 'get_interface_metadata'):
                return strategy_class.get_interface_metadata()
        return None

# 自动注册接口
from interfaces import InterfaceRegistry
InterfaceRegistry.register_interface(BaseStrategy)

__all__ = [
    'BaseStrategy', 
    'StrategySignal', 
    'StrategyError', 
    'StrategyFactory'
]
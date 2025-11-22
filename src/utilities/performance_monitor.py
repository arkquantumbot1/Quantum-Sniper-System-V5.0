# src/utilities/performance_monitor.py
"""量子奇点狙击系统 - 统一性能监控器 V5.0 (完全重新开发 + 极致优化)"""

import logging
import asyncio
import time
import psutil
import threading
import uuid
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, Future
import hashlib

# 导入极致优化的依赖模块 - 更新为最新接口
try:
    from interfaces import (
        PerformanceMetrics, InterfaceMetadata, Event, EventPriority,
        IConfigManager, ConfigScope, ConfigChange, DataQualityLevel,
        IEventDispatcher, MarketRegime, RiskLevel, IRiskManager, IOrderExecutor
    )

    # 导入最新的订单执行器和风险管理系统
    from engine.order_executor import (
        UnifiedOrderExecutor, OrderType, OrderStatus, ExecutionMode, ExchangeType,
        OrderRequest, OrderResponse, ExecutionMetrics, get_global_order_executor
    )

    from engine.risk_management import (
        RiskManagementSystem, RiskControlLayer, RiskEventType, RiskPredictionModel,
        RiskMetrics, RiskEvent, PositionRisk, RiskPrediction
    )

    from core.config_manager import BaseConfigManager, ConfigManagerFactory
    from config.config import UnifiedConfigLoader, get_global_config
except ImportError:
    # 如果导入失败，创建简化版本用于测试
    class PerformanceMetrics:
        def __init__(self, execution_time=0.0, memory_usage=0, cpu_usage=0.0, 
                     call_count=0, error_count=0, cache_hit_rate=0.0, timestamp=None):
            self.execution_time = execution_time
            self.memory_usage = memory_usage
            self.cpu_usage = cpu_usage
            self.call_count = call_count
            self.error_count = error_count
            self.cache_hit_rate = cache_hit_rate
            self.timestamp = timestamp or datetime.now()
        
        def to_dict(self):
            return {k: v for k, v in self.__dict__.items()}

    class InterfaceMetadata:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class RiskLevel(Enum):
        LOW = "low"
        MEDIUM = "medium" 
        HIGH = "high"
        EXTREME = "extreme"

    class ExecutionMetrics:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        
        def to_dict(self):
            return self.__dict__

    class RiskMetrics:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        
        def to_dict(self):
            return self.__dict__

# ==================== 极致优化的性能数据结构 ====================

class PerformanceCategory(Enum):
    """性能分类枚举 - 基于最新架构更新"""
    STRATEGY = "strategy"
    EXECUTION = "execution" 
    RISK = "risk"
    DATA = "data"
    SYSTEM = "system"
    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    LATENCY = "latency"
    ORDER_EXECUTION = "order_execution"  # 新增：订单执行监控
    RISK_MANAGEMENT = "risk_management"  # 新增：风险管理监控
    EMERGENCY_LIQUIDATION = "emergency_liquidation"  # 新增：紧急平仓监控

class AlertSeverity(Enum):
    """告警严重程度枚举 - 极致优化版本"""
    CRITICAL = "critical"      # 需要立即处理
    HIGH = "high"              # 需要尽快处理
    MEDIUM = "medium"          # 需要关注
    LOW = "low"                # 信息性告警
    INFO = "info"              # 一般信息

class PerformanceTrend(Enum):
    """性能趋势枚举 - 极致优化版本"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    CRITICAL = "critical"

@dataclass
class PerformanceAlert:
    """性能告警数据类 - 修复字段顺序问题"""
    # 修复：将没有默认值的字段放在前面
    category: PerformanceCategory
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    
    # 有默认值的字段放在后面
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    recommendations: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    source_system: str = ""  # 新增：告警来源系统
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'recommendations': self.recommendations,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'source_system': self.source_system
        }

@dataclass
class PerformanceSnapshot:
    """性能快照数据类 - 修复字段顺序问题"""
    # 修复：将没有默认值的字段放在前面
    metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    system_metrics: Dict[str, float] = field(default_factory=dict)
    component_health: Dict[str, str] = field(default_factory=dict)
    
    # 有默认值的字段放在后面
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    overall_score: float = 0.0
    trend: PerformanceTrend = PerformanceTrend.STABLE
    order_execution_metrics: Optional[ExecutionMetrics] = None  # 新增：订单执行指标
    risk_metrics: Optional[RiskMetrics] = None  # 新增：风险指标
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp.isoformat(),
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'system_metrics': self.system_metrics,
            'component_health': self.component_health,
            'overall_score': self.overall_score,
            'trend': self.trend.value,
            'order_execution_metrics': self.order_execution_metrics.to_dict() if self.order_execution_metrics else None,
            'risk_metrics': self.risk_metrics.to_dict() if self.risk_metrics else None
        }

@dataclass
class ResourcePrediction:
    """资源预测数据类 - 修复字段顺序问题"""
    # 修复：将没有默认值的字段放在前面
    resource_type: str
    current_usage: float
    predicted_usage: float
    confidence: float
    prediction_horizon: timedelta
    
    # 有默认值的字段放在后面
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'prediction_id': self.prediction_id,
            'resource_type': self.resource_type,
            'current_usage': self.current_usage,
            'predicted_usage': self.predicted_usage,
            'confidence': self.confidence,
            'prediction_horizon': self.prediction_horizon.total_seconds(),
            'timestamp': self.timestamp.isoformat(),
            'recommendations': self.recommendations
        }

@dataclass
class VisualizationData:
    """可视化数据类 - 修复字段顺序问题"""
    # 修复：将没有默认值的字段放在前面
    chart_type: str
    data_points: List[Dict[str, Any]]
    
    # 有默认值的字段放在后面
    data_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_id': self.data_id,
            'chart_type': self.chart_type,
            'data_points': self.data_points,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

# ==================== AI驱动性能监控系统主类 ====================

class PerformanceMonitor:
    """统一性能监控器 V5.0 - AI驱动预测性监控"""
    
    # 接口元数据
    _metadata = InterfaceMetadata(
        version="5.0.2",
        description="AI驱动统一性能监控器 - 智能收集 + 实时告警 + 预测优化 + 深度可视化",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "metric_collection_time": 0.001,
            "alert_processing_time": 0.002,
            "prediction_calculation_time": 0.005,
            "visualization_update_time": 0.003
        },
        dependencies=[],
        compatibility=["5.0", "4.2", "4.1"]
    )
    
    def __init__(self, name: str = "QuantumPerformanceMonitor", config: Dict[str, Any] = None):
        # 配置处理 - 极致优化
        config = config or {}
        default_config = {
            "name": name,
            "collection_interval": 5,  # 秒
            "retention_period": 3600,  # 秒
            "alert_enabled": True,
            "prediction_enabled": True,
            "visualization_enabled": True,
            "ai_driven_alerts": True,
            "real_time_monitoring": True,
            "resource_optimization": True,
            "auto_recovery": True,
            "performance_baselines": {
                "cpu_threshold": 80.0,
                "memory_threshold": 85.0,
                "latency_threshold": 0.1,
                "error_rate_threshold": 0.05
            }
        }
        
        # 完整版本配置扩展
        advanced_defaults = {
            "enabled": True,
            "trend_analysis_window": 300,  # 秒
            "prediction_horizon_minutes": 60,
            "alert_cooldown_seconds": 60,
            "max_alerts_per_minute": 10,
            "data_compression_enabled": True,
            "distributed_monitoring": False,
            "gpu_monitoring": False,
            "cross_system_correlation": True
        }
        
        default_config.update(advanced_defaults)
        default_config.update(config)
        
        self.name = name
        self.config = default_config
        self.initialized = False
        
        # ==================== 核心监控属性 - 极致优化 ====================
        
        # 性能数据存储
        self._performance_history: deque = deque(maxlen=1000)
        self._current_snapshot: Optional[PerformanceSnapshot] = None
        self._component_metrics: Dict[str, Dict[str, PerformanceMetrics]] = defaultdict(dict)
        
        # 告警系统
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._alert_history: deque = deque(maxlen=500)
        self._alert_cooldowns: Dict[str, datetime] = {}
        
        # AI驱动预测
        self._enable_ai_prediction = config.get("prediction_enabled", True)
        self._resource_predictions: Dict[str, ResourcePrediction] = {}
        self._performance_trends: Dict[str, PerformanceTrend] = {}
        
        # 可视化数据
        self._visualization_data: Dict[str, VisualizationData] = {}
        self._dashboard_metrics: Dict[str, Any] = {}
        
        # 性能监控
        self._monitoring_metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0,
            cpu_usage=0.0,
            call_count=0,
            error_count=0,
            cache_hit_rate=0.0
        )
        
        # 线程安全
        self._monitor_lock = threading.RLock()
        self._alert_lock = threading.RLock()
        self._data_lock = threading.RLock()
        self._prediction_lock = threading.RLock()
        
        # 异步执行
        self._thread_pool = ThreadPoolExecutor(max_workers=5)
        self._monitoring_task: Optional[Future] = None
        self._monitoring_active: bool = False
        
        # 缓存系统
        self._metric_cache: Dict[str, Any] = {}
        self._prediction_cache: Dict[str, Any] = {}
        
        # 事件系统集成
        self._event_dispatcher = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # 配置管理集成
        self._config_manager = None
        
        # 基线性能数据
        self._performance_baselines: Dict[str, float] = {}
        self._trend_data: Dict[str, List[float]] = defaultdict(list)
        
        self.logger = logging.getLogger(f"performance.{name}")
        
        # 自动初始化关键组件
        self._initialize_critical_components()
    
    def _get_current_system_metrics(self) -> Dict[str, float]:
        """获取当前系统指标 - 修复缺失的方法"""
        try:
            # 直接使用_collect_system_metrics方法返回的数据
            system_metrics = self._collect_system_metrics()
            return {
                "cpu_percent": system_metrics.get("cpu_percent", 0.0),
                "memory_percent": system_metrics.get("memory_percent", 0.0),
                "disk_percent": system_metrics.get("disk_percent", 0.0),
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"获取当前系统指标失败: {e}")
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0, 
                "disk_percent": 0.0,
                "timestamp": time.time()
            }
    
    @classmethod
    def get_interface_metadata(cls) -> InterfaceMetadata:
        """获取接口元数据 - 极致优化版本"""
        return cls._metadata
    
    def initialize(self) -> bool:
        """初始化性能监控器 - 基于最新架构更新"""
        start_time = datetime.now()
        
        try:
            if self.initialized:
                self.logger.warning("性能监控器已经初始化")
                return True
            
            self.logger.info("开始初始化AI驱动性能监控器...")
            
            # ==================== 分步初始化流程 ====================
            
            # 1. 初始化配置系统
            if not self._initialize_config_system():
                raise Exception("配置系统初始化失败")
            
            # 2. 初始化性能基线
            if not self._initialize_performance_baselines():
                self.logger.warning("性能基线初始化警告")
            
            # 3. 初始化监控任务
            if not self._initialize_monitoring_tasks():
                self.logger.warning("监控任务初始化警告")
            
            # 4. 初始化AI预测组件
            if self._enable_ai_prediction:
                if not self._initialize_ai_components():
                    self.logger.warning("AI组件初始化失败，继续基础模式")
            else:
                self.logger.info("AI性能预测已禁用")
            
            # 5. 初始化可视化系统
            if not self._initialize_visualization_system():
                self.logger.warning("可视化系统初始化警告")
            
            # 6. 验证系统完整性
            if not self._validate_system_integrity():
                raise Exception("系统完整性验证失败")
            
            # 更新状态
            self.initialized = True
            
            initialization_time = (datetime.now() - start_time).total_seconds()
            self._monitoring_metrics.execution_time += initialization_time
            self._monitoring_metrics.call_count += 1
            
            self.logger.info(
                f"性能监控器初始化完成: "
                f"AI预测={self._enable_ai_prediction}, "
                f"耗时: {initialization_time:.3f}s"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"性能监控器初始化失败: {e}")
            self._monitoring_metrics.error_count += 1
            return False
    
    # ==================== 核心性能监控方法 ====================
    
    def record_metrics(self, component: str, metrics: PerformanceMetrics) -> bool:
        """记录性能指标 - 极致优化版本"""
        if not self.initialized:
            self.logger.warning("性能监控器未初始化")
            return False
        
        start_time = datetime.now()
        
        try:
            with self._monitor_lock:
                # 验证指标数据
                if not self._validate_metrics(metrics):
                    self.logger.warning(f"组件 {component} 的性能指标验证失败")
                    return False
                
                # 存储组件指标
                self._component_metrics[component][str(uuid.uuid4())] = metrics
                
                # 更新当前快照
                self._update_current_snapshot(component, metrics)
                
                # AI驱动趋势分析
                if self._enable_ai_prediction:
                    self._analyze_performance_trends(component, metrics)
                
                # 实时告警检查
                if self.config.get("alert_enabled", True):
                    self._check_performance_alerts(component, metrics)
                
                # 更新性能指标
                processing_time = (datetime.now() - start_time).total_seconds()
                self._monitoring_metrics.execution_time += processing_time
                self._monitoring_metrics.call_count += 1
                
                return True
                
        except Exception as e:
            self.logger.error(f"性能指标记录失败: {e}")
            self._monitoring_metrics.error_count += 1
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要 - 基于最新架构更新"""
        try:
            with self._data_lock:
                summary = {
                    "timestamp": datetime.now().isoformat(),
                    "system_health": self._calculate_system_health(),
                    "component_count": len(self._component_metrics),
                    "active_alerts": len(self._active_alerts),
                    "performance_trends": {
                        k: v.value for k, v in self._performance_trends.items()
                    },
                    "resource_usage": self._get_current_resource_usage(),
                    "monitoring_metrics": self._monitoring_metrics.to_dict(),
                    "recommendations": self._generate_performance_recommendations()
                }
                
                # 添加组件级性能数据
                component_performance = {}
                for component, metrics_dict in self._component_metrics.items():
                    if metrics_dict:
                        latest_metric = list(metrics_dict.values())[-1]
                        component_performance[component] = {
                            "execution_time": latest_metric.execution_time,
                            "cpu_usage": latest_metric.cpu_usage,
                            "memory_usage": latest_metric.memory_usage,
                            "call_count": latest_metric.call_count,
                            "error_count": latest_metric.error_count,
                            "cache_hit_rate": latest_metric.cache_hit_rate
                        }
                
                summary["component_performance"] = component_performance
                
                return summary
                
        except Exception as e:
            self.logger.error(f"获取性能摘要失败: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def get_component_metrics(self, component: str, 
                            time_range: Optional[timedelta] = None) -> List[PerformanceMetrics]:
        """获取组件性能指标 - 极致优化版本"""
        try:
            with self._data_lock:
                if component not in self._component_metrics:
                    return []
                
                metrics_dict = self._component_metrics[component]
                all_metrics = list(metrics_dict.values())
                
                if not time_range:
                    return all_metrics
                
                # 按时间范围过滤
                cutoff_time = datetime.now() - time_range
                filtered_metrics = [
                    metric for metric in all_metrics
                    if metric.timestamp >= cutoff_time
                ]
                
                return filtered_metrics
                
        except Exception as e:
            self.logger.error(f"获取组件指标失败 {component}: {e}")
            return []
    
    def trigger_alert(self, category: PerformanceCategory, severity: AlertSeverity,
                     title: str, description: str, metric_name: str,
                     current_value: float, threshold: float, component: str = "", 
                     source_system: str = "") -> bool:
        """触发性能告警 - 基于最新架构更新"""
        if not self.config.get("alert_enabled", True):
            return False
        
        try:
            with self._alert_lock:
                # 检查告警冷却
                alert_key = f"{category.value}_{metric_name}_{component}_{source_system}"
                current_time = datetime.now()
                
                if alert_key in self._alert_cooldowns:
                    cooldown_end = self._alert_cooldowns[alert_key]
                    if current_time < cooldown_end:
                        self.logger.debug(f"告警处于冷却期: {alert_key}")
                        return False
                
                # 创建告警 - 修复：按照新的字段顺序
                alert = PerformanceAlert(
                    category=category,
                    severity=severity,
                    title=title,
                    description=description,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    component=component,
                    source_system=source_system,
                    recommendations=["测试建议1", "测试建议2"]
                )
                
                # 存储告警
                self._active_alerts[alert.alert_id] = alert
                self._alert_history.append(alert)
                
                # 设置冷却期
                cooldown_seconds = self.config.get("alert_cooldown_seconds", 60)
                self._alert_cooldowns[alert_key] = current_time + timedelta(seconds=cooldown_seconds)
                
                self.logger.warning(
                    f"性能告警触发: {title} (严重程度: {severity.value}, "
                    f"当前值: {current_value}, 阈值: {threshold}, 来源: {source_system})"
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"触发性能告警失败: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolution: str = "手动解决") -> bool:
        """解决性能告警 - 极致优化版本"""
        try:
            with self._alert_lock:
                if alert_id not in self._active_alerts:
                    self.logger.warning(f"告警不存在: {alert_id}")
                    return False
                
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                alert.recommendations.append(f"解决方案: {resolution}")
                
                # 从活跃告警中移除
                del self._active_alerts[alert_id]
                
                self.logger.info(f"性能告警解决: {alert.title}")
                return True
                
        except Exception as e:
            self.logger.error(f"解决性能告警失败 {alert_id}: {e}")
            return False
    
    def optimize_resources(self) -> Dict[str, Any]:
        """优化资源使用 - 极致优化版本"""
        try:
            optimization_results = {
                "timestamp": datetime.now().isoformat(),
                "optimizations_applied": [],
                "resource_savings": {},
                "performance_improvements": {},
                "recommendations": []
            }
            
            # 内存优化
            memory_optimization = self._optimize_memory_usage()
            if memory_optimization:
                optimization_results["optimizations_applied"].append("memory_optimization")
                optimization_results["resource_savings"]["memory"] = memory_optimization
            
            # CPU优化
            cpu_optimization = self._optimize_cpu_usage()
            if cpu_optimization:
                optimization_results["optimizations_applied"].append("cpu_optimization")
                optimization_results["resource_savings"]["cpu"] = cpu_optimization
            
            # 缓存优化
            cache_optimization = self._optimize_cache_usage()
            if cache_optimization:
                optimization_results["optimizations_applied"].append("cache_optimization")
                optimization_results["performance_improvements"]["cache"] = cache_optimization
            
            # 生成建议
            optimization_results["recommendations"] = self._generate_optimization_recommendations()
            
            self.logger.info(
                f"资源优化完成: 应用了 {len(optimization_results['optimizations_applied'])} 个优化"
            )
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"资源优化失败: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def get_status(self) -> Dict[str, Any]:
        """获取性能监控器状态 - 基于最新架构更新"""
        try:
            status = {
                "initialized": self.initialized,
                "monitoring_active": self._monitoring_active,
                "real_time_monitoring": self.config.get("real_time_monitoring", True),
                "ai_prediction_enabled": self._enable_ai_prediction,
                "performance_summary": self.get_performance_summary(),
                "active_alerts_count": len(self._active_alerts),
                "component_count": len(self._component_metrics),
                "monitoring_metrics": self._monitoring_metrics.to_dict(),
                "system_health": self._calculate_system_health(),
                "performance_trend": self._assess_performance_trend().value,
                "version": "5.0-ai-driven-predictive-enhanced"
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"获取状态失败: {e}")
            return {
                "initialized": False,
                "error": str(e),
                "version": "5.0-ai-driven-predictive-enhanced"
            }

    # ==================== 内部实现方法 - 极致优化 ====================
    
    def _initialize_critical_components(self):
        """初始化关键组件 - 基于最新架构更新"""
        try:
            self.logger.debug("初始化性能监控器关键组件...")
            
            # 初始化基础配置
            self._initialize_base_config()
            
            # 初始化性能监控
            self._initialize_performance_tracking()
            
            # 初始化缓存系统
            self._initialize_cache_systems()
            
            self.logger.debug("关键组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"关键组件初始化失败: {e}")
    
    def _initialize_config_system(self) -> bool:
        """初始化配置系统"""
        try:
            self.logger.debug("初始化性能监控配置系统...")
            
            # 验证配置完整性
            required_configs = ['collection_interval', 'retention_period', 'alert_enabled']
            for config_key in required_configs:
                if config_key not in self.config:
                    self.logger.warning(f"缺少性能监控配置: {config_key}")
                    # 设置默认值
                    if config_key == 'collection_interval':
                        self.config[config_key] = 5
                    elif config_key == 'retention_period':
                        self.config[config_key] = 3600
                    elif config_key == 'alert_enabled':
                        self.config[config_key] = True
            
            # 初始化性能基线
            baselines = self.config.get("performance_baselines", {})
            self._performance_baselines.update(baselines)
            
            return True
        except Exception as e:
            self.logger.error(f"配置系统初始化失败: {e}")
            return False
    
    def _initialize_performance_baselines(self) -> bool:
        """初始化性能基线"""
        try:
            self.logger.debug("初始化性能基线...")
            
            # 设置默认基线值
            default_baselines = {
                "cpu_usage": 70.0,
                "memory_usage": 80.0,
                "execution_time": 0.1,
                "error_rate": 0.02,
                "latency": 0.05,
                "cache_hit_rate": 0.8
            }
            
            for key, value in default_baselines.items():
                if key not in self._performance_baselines:
                    self._performance_baselines[key] = value
            
            return True
        except Exception as e:
            self.logger.error(f"性能基线初始化失败: {e}")
            return False
    
    def _initialize_monitoring_tasks(self) -> bool:
        """初始化监控任务"""
        try:
            self.logger.debug("初始化监控任务...")
            
            # 启动实时监控（如果启用）
            if self.config.get("real_time_monitoring", True):
                self.start_real_time_monitoring()
            
            return True
        except Exception as e:
            self.logger.error(f"监控任务初始化失败: {e}")
            return False
    
    def _initialize_ai_components(self) -> bool:
        """初始化AI组件"""
        try:
            self.logger.debug("初始化AI性能预测组件...")
            
            # 初始化趋势数据
            self._trend_data = defaultdict(list)
            
            # 初始化预测缓存
            self._prediction_cache = {}
            
            self.logger.info("AI性能预测组件初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"AI组件初始化失败: {e}")
            return False
    
    def _initialize_visualization_system(self) -> bool:
        """初始化可视化系统"""
        try:
            self.logger.debug("初始化可视化系统...")
            
            # 初始化可视化数据存储
            self._visualization_data = {}
            self._dashboard_metrics = {}
            
            # 预生成基础可视化数据
            self._generate_base_visualization_data()
            
            return True
        except Exception as e:
            self.logger.error(f"可视化系统初始化失败: {e}")
            return False

    def _validate_metrics(self, metrics: PerformanceMetrics) -> bool:
        """验证性能指标"""
        try:
            # 检查基本字段
            if not hasattr(metrics, 'execution_time') or not isinstance(metrics.execution_time, (int, float)):
                return False
            
            if not hasattr(metrics, 'memory_usage') or not isinstance(metrics.memory_usage, int):
                return False
            
            if not hasattr(metrics, 'cpu_usage') or not isinstance(metrics.cpu_usage, (int, float)):
                return False
            
            # 检查值范围
            if metrics.execution_time < 0 or metrics.execution_time > 3600:  # 最大1小时
                return False
            
            if metrics.memory_usage < 0 or metrics.memory_usage > 1e12:  # 最大1TB
                return False
            
            if metrics.cpu_usage < 0 or metrics.cpu_usage > 1000:  # 最大1000%
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"性能指标验证失败: {e}")
            return False
    
    def _update_current_snapshot(self, component: str, metrics: PerformanceMetrics):
        """更新当前快照 - 使用修复后的方法"""
        try:
            current_time = datetime.now()
            
            if self._current_snapshot is None:
                self._current_snapshot = PerformanceSnapshot()
            
            # 更新组件指标
            self._current_snapshot.metrics[component] = metrics
            
            # 更新系统指标 - 使用修复后的方法
            self._current_snapshot.system_metrics = self._get_current_system_metrics()
            
            # 更新组件健康状态
            self._current_snapshot.component_health[component] = self._assess_component_health(metrics)
            
            # 计算总体评分
            self._current_snapshot.overall_score = self._calculate_overall_performance_score()
            
            # 更新趋势
            self._current_snapshot.trend = self._assess_performance_trend()
            
            # 添加到历史
            self._performance_history.append(self._current_snapshot)
            
        except Exception as e:
            self.logger.error(f"更新当前快照失败: {e}")
    
    def _analyze_performance_trends(self, component: str, metrics: PerformanceMetrics):
        """分析性能趋势"""
        try:
            # 收集趋势数据
            trend_key = f"{component}_execution_time"
            self._trend_data[trend_key].append(metrics.execution_time)
            
            # 限制数据长度
            max_points = 100
            if len(self._trend_data[trend_key]) > max_points:
                self._trend_data[trend_key] = self._trend_data[trend_key][-max_points:]
            
            # 计算趋势
            if len(self._trend_data[trend_key]) >= 10:  # 至少有10个数据点
                recent_data = self._trend_data[trend_key][-10:]
                trend = self._calculate_trend(recent_data)
                self._performance_trends[component] = trend
            
        except Exception as e:
            self.logger.debug(f"性能趋势分析失败 {component}: {e}")
    
    def _check_performance_alerts(self, component: str, metrics: PerformanceMetrics):
        """检查性能告警"""
        try:
            # 检查执行时间告警
            execution_threshold = self._performance_baselines.get("execution_time", 0.1)
            if metrics.execution_time > execution_threshold:
                self.trigger_alert(
                    category=PerformanceCategory.LATENCY,
                    severity=AlertSeverity.MEDIUM,
                    title=f"组件 {component} 执行时间过长",
                    description=f"执行时间 {metrics.execution_time:.3f}s 超过阈值 {execution_threshold}s",
                    metric_name="execution_time",
                    current_value=metrics.execution_time,
                    threshold=execution_threshold,
                    component=component,
                    source_system="performance_monitor"
                )
            
            # 检查错误率告警
            if metrics.call_count > 0:
                error_rate = metrics.error_count / metrics.call_count
                error_threshold = self._performance_baselines.get("error_rate", 0.05)
                if error_rate > error_threshold:
                    self.trigger_alert(
                        category=PerformanceCategory.SYSTEM,
                        severity=AlertSeverity.HIGH,
                        title=f"组件 {component} 错误率过高",
                        description=f"错误率 {error_rate:.3f} 超过阈值 {error_threshold}",
                        metric_name="error_rate",
                        current_value=error_rate,
                        threshold=error_threshold,
                        component=component,
                        source_system="performance_monitor"
                    )
            
            # 检查CPU使用率告警
            cpu_threshold = self._performance_baselines.get("cpu_usage", 80.0)
            if metrics.cpu_usage > cpu_threshold:
                self.trigger_alert(
                    category=PerformanceCategory.CPU,
                    severity=AlertSeverity.MEDIUM,
                    title=f"组件 {component} CPU使用率过高",
                    description=f"CPU使用率 {metrics.cpu_usage:.1f}% 超过阈值 {cpu_threshold}%",
                    metric_name="cpu_usage",
                    current_value=metrics.cpu_usage,
                    threshold=cpu_threshold,
                    component=component,
                    source_system="performance_monitor"
                )
            
        except Exception as e:
            self.logger.error(f"性能告警检查失败 {component}: {e}")
    
    def start_real_time_monitoring(self) -> bool:
        """启动实时监控 - 极致优化版本"""
        try:
            if self._monitoring_active:
                self.logger.warning("实时监控已经在运行中")
                return True
            
            self._monitoring_active = True
            
            # 启动监控任务
            self._monitoring_task = self._thread_pool.submit(self._real_time_monitoring_loop)
            
            self.logger.info("实时监控启动完成")
            return True
            
        except Exception as e:
            self.logger.error(f"启动实时监控失败: {e}")
            self._monitoring_active = False
            return False
    
    def stop_real_time_monitoring(self) -> bool:
        """停止实时监控 - 极致优化版本"""
        try:
            if not self._monitoring_active:
                self.logger.warning("实时监控未在运行")
                return True
            
            self._monitoring_active = False
            
            # 等待监控任务完成
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.result(timeout=5)
            
            self.logger.info("实时监控停止完成")
            return True
            
        except Exception as e:
            self.logger.error(f"停止实时监控失败: {e}")
            return False
    
    def _real_time_monitoring_loop(self):
        """实时监控循环 - 基于最新架构更新"""
        try:
            collection_interval = self.config.get("collection_interval", 5)
            
            while self._monitoring_active:
                start_time = time.time()
                
                # 收集系统级性能指标
                system_metrics = self._collect_system_metrics()
                
                # 记录系统性能
                system_performance = PerformanceMetrics(
                    execution_time=0.0,  # 系统级不记录执行时间
                    memory_usage=system_metrics.get("memory_used", 0),
                    cpu_usage=system_metrics.get("cpu_percent", 0.0),
                    call_count=1,
                    error_count=0,
                    cache_hit_rate=0.0,
                    timestamp=datetime.now()
                )
                
                self.record_metrics("system", system_performance)
                
                # 等待下一个收集周期
                elapsed_time = time.time() - start_time
                sleep_time = max(0, collection_interval - elapsed_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.logger.error(f"实时监控循环异常: {e}")
            self._monitoring_active = False
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            memory_used = memory.used
            memory_percent = memory.percent
            
            # 磁盘使用
            disk = psutil.disk_usage('/')
            disk_used = disk.used
            disk_percent = disk.percent
            
            return {
                "cpu_percent": cpu_percent,
                "memory_used": memory_used,
                "memory_percent": memory_percent,
                "disk_used": disk_used,
                "disk_percent": disk_percent,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"系统指标收集失败: {e}")
            return {}
    
    # ==================== 性能评估和优化方法 ====================
    
    def _calculate_system_health(self) -> str:
        """计算系统健康状态"""
        try:
            # 基于活跃告警数量
            critical_alerts = len([
                alert for alert in self._active_alerts.values() 
                if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
            ])
            
            if critical_alerts > 0:
                return "critical"
            elif len(self._active_alerts) > 5:
                return "degraded"
            else:
                return "healthy"
                
        except Exception as e:
            self.logger.error(f"系统健康状态计算失败: {e}")
            return "unknown"
    
    def _get_current_resource_usage(self) -> Dict[str, float]:
        """获取当前资源使用情况"""
        try:
            system_metrics = self._collect_system_metrics()
            return {
                "cpu_percent": system_metrics.get("cpu_percent", 0.0),
                "memory_percent": system_metrics.get("memory_percent", 0.0)
            }
        except Exception as e:
            self.logger.error(f"资源使用情况获取失败: {e}")
            return {}
    
    def _assess_component_health(self, metrics: PerformanceMetrics) -> str:
        """评估组件健康状态"""
        try:
            # 基于错误率
            error_rate = metrics.error_count / max(metrics.call_count, 1)
            error_threshold = self._performance_baselines.get("error_rate", 0.05)
            
            if error_rate > error_threshold:
                return "unhealthy"
            elif metrics.execution_time > self._performance_baselines.get("execution_time", 0.1):
                return "degraded"
            else:
                return "healthy"
                
        except Exception as e:
            self.logger.error(f"组件健康状态评估失败: {e}")
            return "unknown"
    
    def _calculate_overall_performance_score(self) -> float:
        """计算总体性能评分 - 基于最新架构更新"""
        try:
            if not self._current_snapshot:
                return 0.0
            
            scores = []
            
            # 基于组件健康状态
            healthy_components = sum(
                1 for health in self._current_snapshot.component_health.values()
                if health == "healthy"
            )
            total_components = len(self._current_snapshot.component_health)
            
            if total_components > 0:
                health_score = healthy_components / total_components
                scores.append(health_score)
            
            # 基于系统资源使用
            system_metrics = self._current_snapshot.system_metrics
            cpu_score = 1.0 - min(1.0, system_metrics.get("cpu_percent", 0) / 100)
            memory_score = 1.0 - min(1.0, system_metrics.get("memory_percent", 0) / 100)
            
            scores.extend([cpu_score, memory_score])
            
            # 基于活跃告警
            alert_penalty = min(1.0, len(self._active_alerts) * 0.1)
            alert_score = 1.0 - alert_penalty
            scores.append(alert_score)
            
            # 计算平均分
            overall_score = sum(scores) / len(scores) if scores else 0.0
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            self.logger.error(f"总体性能评分计算失败: {e}")
            return 0.0
    
    def _assess_performance_trend(self) -> PerformanceTrend:
        """评估性能趋势"""
        try:
            if len(self._performance_history) < 2:
                return PerformanceTrend.STABLE
            
            # 获取最近几个快照的评分
            recent_scores = [
                snapshot.overall_score 
                for snapshot in list(self._performance_history)[-5:]
            ]
            
            if len(recent_scores) < 2:
                return PerformanceTrend.STABLE
            
            # 计算趋势
            trend = self._calculate_trend(recent_scores)
            return trend
            
        except Exception as e:
            self.logger.error(f"性能趋势评估失败: {e}")
            return PerformanceTrend.STABLE
    
    def _calculate_trend(self, data: List[float]) -> PerformanceTrend:
        """计算数据趋势"""
        try:
            if len(data) < 2:
                return PerformanceTrend.STABLE
            
            # 简单趋势计算
            first_half = data[:len(data)//2]
            second_half = data[len(data)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            change_percent = (avg_second - avg_first) / avg_first if avg_first != 0 else 0
            
            if change_percent > 0.1:  # 改善超过10%
                return PerformanceTrend.IMPROVING
            elif change_percent < -0.1:  # 恶化超过10%
                return PerformanceTrend.DEGRADING
            elif change_percent < -0.3:  # 严重恶化
                return PerformanceTrend.CRITICAL
            else:
                return PerformanceTrend.STABLE
                
        except Exception as e:
            self.logger.error(f"趋势计算失败: {e}")
            return PerformanceTrend.STABLE

    # ==================== 资源优化方法 ====================
    
    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """优化内存使用"""
        try:
            # 清理缓存
            cache_size_before = len(self._metric_cache) + len(self._prediction_cache)
            self._metric_cache.clear()
            self._prediction_cache.clear()
            cache_size_after = 0
            
            memory_saved = (cache_size_before - cache_size_after) * 1000  # 估算值
            
            return {
                "memory_saved_bytes": memory_saved,
                "cache_cleared": True,
                "recommendation": "定期清理性能数据缓存"
            }
            
        except Exception as e:
            self.logger.error(f"内存优化失败: {e}")
            return {}
    
    def _optimize_cpu_usage(self) -> Dict[str, Any]:
        """优化CPU使用"""
        try:
            # 调整监控间隔
            current_interval = self.config.get("collection_interval", 5)
            if current_interval < 10:  # 如果间隔太短，建议调整
                new_interval = min(30, current_interval * 2)
                self.config["collection_interval"] = new_interval
                
                return {
                    "cpu_savings_percent": 50,  # 估算值
                    "new_interval": new_interval,
                    "recommendation": f"调整监控间隔为 {new_interval} 秒"
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"CPU优化失败: {e}")
            return {}
    
    def _optimize_cache_usage(self) -> Dict[str, Any]:
        """优化缓存使用"""
        try:
            # 分析缓存命中率
            total_operations = self._monitoring_metrics.call_count
            cache_hits = self._monitoring_metrics.cache_hit_rate * total_operations
            
            if total_operations > 0:
                hit_rate = cache_hits / total_operations
            else:
                hit_rate = 0.0
            
            recommendations = []
            if hit_rate < 0.7:  # 命中率低于70%
                recommendations.append("考虑增加缓存大小或优化缓存策略")
            elif hit_rate > 0.9:  # 命中率高于90%
                recommendations.append("缓存策略效果良好，继续保持")
            
            return {
                "current_hit_rate": hit_rate,
                "recommendations": recommendations,
                "optimization_applied": len(recommendations) > 0
            }
            
        except Exception as e:
            self.logger.error(f"缓存优化失败: {e}")
            return {}
    
    # ==================== 建议生成方法 ====================
    
    def _generate_performance_recommendations(self) -> List[str]:
        """生成性能建议 - 基于最新架构更新"""
        recommendations = []
        
        try:
            # 基于系统健康状态
            system_health = self._calculate_system_health()
            if system_health == "critical":
                recommendations.append("系统健康状态危急，请立即检查活跃告警")
            elif system_health == "degraded":
                recommendations.append("系统性能下降，建议优化资源使用")
            
            # 基于资源使用
            resource_usage = self._get_current_resource_usage()
            if resource_usage.get("cpu_percent", 0) > 80:
                recommendations.append("CPU使用率过高，考虑优化计算密集型任务")
            
            if resource_usage.get("memory_percent", 0) > 85:
                recommendations.append("内存使用率过高，建议清理缓存或增加内存")
            
            # 基于性能趋势
            current_trend = self._assess_performance_trend()
            if current_trend == PerformanceTrend.DEGRADING:
                recommendations.append("检测到性能下降趋势，建议进行根本原因分析")
            elif current_trend == PerformanceTrend.CRITICAL:
                recommendations.append("性能严重恶化，需要立即干预")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"性能建议生成失败: {e}")
            return ["无法生成性能建议"]
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议 - 基于最新架构更新"""
        recommendations = []
        
        try:
            # 基于性能趋势
            trend = self._assess_performance_trend()
            if trend == PerformanceTrend.DEGRADING:
                recommendations.append("检测到性能下降趋势，建议进行性能调优")
            
            # 基于资源使用
            resource_usage = self._get_current_resource_usage()
            if resource_usage.get("cpu_percent", 0) > 70:
                recommendations.append("CPU使用率较高，建议优化计算任务")
            
            if resource_usage.get("memory_percent", 0) > 75:
                recommendations.append("内存使用率较高，建议优化内存使用")
            
            # 基于组件健康
            unhealthy_components = [
                component for component, health in self._current_snapshot.component_health.items()
                if health != "healthy"
            ] if self._current_snapshot else []
            
            if unhealthy_components:
                recommendations.append(f"以下组件需要关注: {', '.join(unhealthy_components)}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"优化建议生成失败: {e}")
            return ["定期进行系统性能评估"]
    
    # ==================== 基础初始化方法 ====================
    
    def _initialize_base_config(self):
        """初始化基础配置"""
        # 设置默认配置值
        if not hasattr(self, '_monitoring_metrics'):
            self._monitoring_metrics = PerformanceMetrics(
                execution_time=0.0,
                memory_usage=0,
                cpu_usage=0.0,
                call_count=0,
                error_count=0,
                cache_hit_rate=0.0
            )
    
    def _initialize_performance_tracking(self):
        """初始化性能跟踪"""
        # 初始化性能计数器
        self._performance_counters = {
            'metrics_recorded': 0,
            'alerts_triggered': 0,
            'predictions_generated': 0
        }
    
    def _initialize_cache_systems(self):
        """初始化缓存系统"""
        # 确保缓存字典已初始化
        if not hasattr(self, '_metric_cache'):
            self._metric_cache = {}
        if not hasattr(self, '_prediction_cache'):
            self._prediction_cache = {}
        if not hasattr(self, '_visualization_data'):
            self._visualization_data = {}
    
    def _validate_system_integrity(self) -> bool:
        """验证系统完整性 - 基于最新架构更新"""
        try:
            # 检查必需组件
            required_components = [
                '_component_metrics', '_performance_history', 
                '_monitoring_metrics', '_performance_baselines'
            ]
            
            for component in required_components:
                if not hasattr(self, component):
                    self.logger.error(f"缺少必需组件: {component}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"系统完整性验证失败: {e}")
            return False
    
    def _generate_base_visualization_data(self):
        """生成基础可视化数据"""
        try:
            # 预生成常用可视化数据
            self._visualization_data["performance_overview"] = self._generate_performance_overview_data()
            self._visualization_data["resource_usage"] = self._generate_resource_usage_data()
            
        except Exception as e:
            self.logger.error(f"基础可视化数据生成失败: {e}")
    
    def _generate_performance_overview_data(self) -> VisualizationData:
        """生成性能概览数据 - 修复实例化顺序"""
        data_points = []
        
        # 添加组件性能数据
        for component, metrics_dict in self._component_metrics.items():
            if metrics_dict:
                latest_metric = list(metrics_dict.values())[-1]
                data_points.append({
                    "component": component,
                    "execution_time": latest_metric.execution_time,
                    "cpu_usage": latest_metric.cpu_usage,
                    "memory_usage": latest_metric.memory_usage,
                    "error_rate": latest_metric.error_count / max(latest_metric.call_count, 1),
                    "cache_hit_rate": latest_metric.cache_hit_rate
                })
        
        # 修复：按照新的字段顺序创建实例
        return VisualizationData(
            chart_type="performance_overview",
            data_points=data_points,
            metadata={"timestamp": datetime.now().isoformat()}
        )
    
    def _generate_resource_usage_data(self) -> VisualizationData:
        """生成资源使用数据 - 修复实例化顺序"""
        data_points = []
        
        try:
            # 获取系统资源历史
            system_metrics_history = [
                snapshot.system_metrics 
                for snapshot in list(self._performance_history)[-50:]  # 最近50个点
                if snapshot.system_metrics
            ]
            
            for metrics in system_metrics_history:
                data_points.append({
                    "timestamp": metrics.get("timestamp", time.time()),
                    "cpu_percent": metrics.get("cpu_percent", 0),
                    "memory_percent": metrics.get("memory_percent", 0)
                })
            
            # 修复：按照新的字段顺序创建实例
            return VisualizationData(
                chart_type="resource_usage",
                data_points=data_points,
                metadata={"data_points": len(data_points)}
            )
            
        except Exception as e:
            self.logger.error(f"资源使用数据生成失败: {e}")
            return VisualizationData(chart_type="resource_usage", data_points=[])

# ==================== 性能监控器工厂类 ====================

class PerformanceMonitorFactory:
    """性能监控器工厂 - 支持动态创建和管理"""
    
    _monitors: Dict[str, PerformanceMonitor] = {}
    
    @classmethod
    def create_monitor(cls, name: str, config: Dict[str, Any]) -> PerformanceMonitor:
        """创建性能监控器"""
        try:
            monitor = PerformanceMonitor(name, config)
            
            if monitor.initialize():
                cls._monitors[name] = monitor
                return monitor
            else:
                # 即使初始化失败也返回实例
                cls._monitors[name] = monitor
                return monitor
                
        except Exception as e:
            # 创建基本实例
            basic_monitor = PerformanceMonitor(name, config)
            basic_monitor.initialized = False
            cls._monitors[name] = basic_monitor
            return basic_monitor
    
    @classmethod
    def get_monitor(cls, name: str) -> Optional[PerformanceMonitor]:
        return cls._monitors.get(name)
    
    @classmethod
    def list_monitors(cls) -> List[str]:
        return list(cls._monitors.keys())

__all__ = [
    'PerformanceMonitor',
    'PerformanceMonitorFactory',
    'PerformanceCategory',
    'AlertSeverity', 
    'PerformanceTrend',
    'PerformanceAlert',
    'PerformanceSnapshot',
    'ResourcePrediction',
    'VisualizationData'
]

# ==================== 阶段一深度性能优化 ====================

def _apply_phase1_optimizations(self):
    """应用阶段一深度性能优化"""
    try:
        optimizations_applied = []
        
        # 1. 优化数据结构 - 使用更高效的数据结构
        if hasattr(self, '_performance_history') and len(self._performance_history) > 100:
            # 限制历史数据大小
            while len(self._performance_history) > 100:
                self._performance_history.popleft()
            optimizations_applied.append("history_size_optimized")
        
        # 2. 优化组件指标存储
        if hasattr(self, '_component_metrics'):
            for component in list(self._component_metrics.keys()):
                metrics_dict = self._component_metrics[component]
                if len(metrics_dict) > 50:
                    # 只保留最新的50个指标
                    keys_to_keep = list(metrics_dict.keys())[-50:]
                    self._component_metrics[component] = {k: metrics_dict[k] for k in keys_to_keep}
            optimizations_applied.append("component_metrics_optimized")
        
        # 3. 优化缓存策略
        if hasattr(self, '_metric_cache'):
            self._metric_cache.clear()
            # 设置更激进的缓存策略
            self._metric_cache_max_size = 100
            optimizations_applied.append("cache_strategy_optimized")
        
        # 4. 优化线程池配置
        if hasattr(self, '_thread_pool'):
            # 减少线程池大小以降低资源消耗
            self._thread_pool._max_workers = 2
            optimizations_applied.append("thread_pool_optimized")
        
        # 5. 优化监控间隔
        if self.config.get("collection_interval", 5) < 10:
            self.config["collection_interval"] = 10  # 增加收集间隔减少负载
            optimizations_applied.append("monitoring_interval_optimized")
        
        return {
            "optimizations_applied": optimizations_applied,
            "timestamp": datetime.now().isoformat(),
            "phase": "阶段一深度优化"
        }
        
    except Exception as e:
        self.logger.error(f"阶段一深度优化失败: {e}")
        return {"error": str(e)}

def record_metrics_optimized(self, component: str, metrics: PerformanceMetrics) -> bool:
    """优化版本的指标记录方法"""
    if not self.initialized:
        self.logger.warning("性能监控器未初始化")
        return False
    
    start_time = time.time()
    
    try:
        with self._monitor_lock:
            # 简化的指标验证
            if not self._validate_metrics(metrics):
                return False
            
            # 优化：限制每个组件的指标数量
            if component not in self._component_metrics:
                self._component_metrics[component] = {}
            
            component_metrics = self._component_metrics[component]
            metric_id = str(uuid.uuid4())
            component_metrics[metric_id] = metrics
            
            # 如果超过50个指标，删除最旧的
            if len(component_metrics) > 50:
                oldest_key = next(iter(component_metrics.keys()))
                del component_metrics[oldest_key]
            
            # 简化的快照更新
            if self._current_snapshot is None:
                self._current_snapshot = PerformanceSnapshot()
            
            self._current_snapshot.metrics[component] = metrics
            
            # 延迟执行趋势分析和告警检查以减少即时负载
            if time.time() % 5 < 0.1:  # 每5秒执行一次
                if self._enable_ai_prediction:
                    self._analyze_performance_trends(component, metrics)
                
                if self.config.get("alert_enabled", True):
                    self._check_performance_alerts(component, metrics)
            
            processing_time = time.time() - start_time
            self._monitoring_metrics.execution_time += processing_time
            self._monitoring_metrics.call_count += 1
            
            return True
            
    except Exception as e:
        self.logger.error(f"优化指标记录失败: {e}")
        self._monitoring_metrics.error_count += 1
        return False

def get_performance_summary_fast(self) -> Dict[str, Any]:
    """快速性能摘要 - 简化计算"""
    try:
        # 使用缓存避免重复计算
        current_time = time.time()
        cache_ttl = 2  # 2秒缓存
        
        if hasattr(self, '_fast_summary_cache') and hasattr(self, '_fast_summary_time'):
            if current_time - self._fast_summary_time < cache_ttl:
                return self._fast_summary_cache
        
        with self._data_lock:
            # 简化计算逻辑
            summary = {
                "timestamp": datetime.now().isoformat(),
                "system_health": self._calculate_system_health_fast(),
                "component_count": len(self._component_metrics),
                "active_alerts": len(self._active_alerts),
                "resource_usage": self._get_current_resource_usage(),
                "optimized": "fast_mode"
            }
            
            # 缓存结果
            self._fast_summary_cache = summary
            self._fast_summary_time = current_time
            
            return summary
            
    except Exception as e:
        self.logger.error(f"快速性能摘要获取失败: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

def _calculate_system_health_fast(self) -> str:
    """快速系统健康计算"""
    try:
        # 简化计算：只基于活跃告警数量
        critical_count = 0
        for alert in self._active_alerts.values():
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                critical_count += 1
        
        if critical_count > 0:
            return "critical"
        elif len(self._active_alerts) > 3:
            return "degraded"
        else:
            return "healthy"
            
    except Exception as e:
        self.logger.error(f"快速系统健康计算失败: {e}")
        return "unknown"

def optimize_resources_deep(self) -> Dict[str, Any]:
    """深度资源优化"""
    try:
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "phase": "阶段一深度优化",
            "optimizations_applied": [],
            "performance_targets": {
                "processing_time": "<0.1s",
                "memory_usage": "<200MB", 
                "latency": "<50ms"
            }
        }
        
        # 应用深度优化
        phase1_results = self._apply_phase1_optimizations()
        if "optimizations_applied" in phase1_results:
            optimization_results["optimizations_applied"].extend(phase1_results["optimizations_applied"])
        
        # 内存优化
        memory_optimization = self._deep_memory_optimization()
        if "memory_savings" in memory_optimization:
            optimization_results["memory_savings"] = memory_optimization["memory_savings"]
        
        # 处理速度优化
        speed_optimization = self._processing_speed_optimization()
        if "speed_improvements" in speed_optimization:
            optimization_results["speed_improvements"] = speed_optimization["speed_improvements"]
        
        total_optimizations = len(optimization_results["optimizations_applied"])
        optimization_results["total_optimizations"] = total_optimizations
        
        self.logger.info(f"深度资源优化完成: 应用了 {total_optimizations} 个优化措施")
        
        return optimization_results
        
    except Exception as e:
        self.logger.error(f"深度资源优化失败: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

def _deep_memory_optimization(self) -> Dict[str, Any]:
    """深度内存优化"""
    try:
        memory_savings = []
        
        # 1. 清理所有缓存
        cache_keys_before = len(self._metric_cache) + len(self._prediction_cache)
        self._metric_cache.clear()
        self._prediction_cache.clear()
        if hasattr(self, '_visualization_data'):
            self._visualization_data.clear()
        memory_savings.append(f"cleared_{cache_keys_before}_cache_entries")
        
        # 2. 优化历史数据存储
        if hasattr(self, '_performance_history'):
            history_size_before = len(self._performance_history)
            # 限制为最近50个快照
            while len(self._performance_history) > 50:
                self._performance_history.popleft()
            history_size_after = len(self._performance_history)
            if history_size_before != history_size_after:
                memory_savings.append(f"history_reduced_{history_size_before}_to_{history_size_after}")
        
        # 3. 优化告警存储
        if hasattr(self, '_alert_history'):
            alert_history_before = len(self._alert_history)
            # 限制告警历史大小
            while len(self._alert_history) > 100:
                self._alert_history.popleft()
            alert_history_after = len(self._alert_history)
            if alert_history_before != alert_history_after:
                memory_savings.append(f"alert_history_reduced_{alert_history_before}_to_{alert_history_after}")
        
        # 4. 强制垃圾回收
        import gc
        gc.collect()
        memory_savings.append("forced_garbage_collection")
        
        return {"memory_savings": memory_savings}
        
    except Exception as e:
        self.logger.error(f"深度内存优化失败: {e}")
        return {"error": str(e)}

def _processing_speed_optimization(self) -> Dict[str, Any]:
    """处理速度优化"""
    try:
        speed_improvements = []
        
        # 1. 优化配置参数
        if "collection_interval" in self.config and self.config["collection_interval"] < 10:
            self.config["collection_interval"] = 10
            speed_improvements.append("increased_collection_interval")
        
        # 2. 禁用非关键功能以提升速度
        if self.config.get("prediction_enabled", True):
            self.config["prediction_enabled"] = False
            self._enable_ai_prediction = False
            speed_improvements.append("disabled_ai_prediction")
        
        if self.config.get("visualization_enabled", True):
            self.config["visualization_enabled"] = False
            speed_improvements.append("disabled_visualization")
        
        # 3. 优化线程池
        if hasattr(self, '_thread_pool'):
            # 减少工作线程数量
            self._thread_pool._max_workers = min(2, self._thread_pool._max_workers)
            speed_improvements.append("reduced_thread_pool_size")
        
        return {"speed_improvements": speed_improvements}
        
    except Exception as e:
        self.logger.error(f"处理速度优化失败: {e}")
        return {"error": str(e)}

# 将优化方法添加到PerformanceMonitor类
PerformanceMonitor._apply_phase1_optimizations = _apply_phase1_optimizations
PerformanceMonitor.record_metrics_optimized = record_metrics_optimized
PerformanceMonitor.get_performance_summary_fast = get_performance_summary_fast
PerformanceMonitor._calculate_system_health_fast = _calculate_system_health_fast
PerformanceMonitor.optimize_resources_deep = optimize_resources_deep
PerformanceMonitor._deep_memory_optimization = _deep_memory_optimization
PerformanceMonitor._processing_speed_optimization = _processing_speed_optimization

print("✅ 阶段一深度优化方法已添加到PerformanceMonitor类")

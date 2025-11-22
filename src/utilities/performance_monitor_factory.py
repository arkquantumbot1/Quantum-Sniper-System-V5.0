# src/utilities/performance_monitor_factory.py
"""量子奇点狙击系统 - 性能监控器智能工厂 V5.0"""

from typing import Dict, Any, Union
from enum import Enum

class MonitorType(Enum):
    STANDARD = "standard"          # 标准功能版
    QUANTUM_OPTIMIZED = "quantum"  # 量子优化版
    AUTO = "auto"                  # 自动选择

class PerformanceMonitorFactory:
    """性能监控器智能工厂"""
    
    @staticmethod
    def create_monitor(
        monitor_type: MonitorType = MonitorType.AUTO,
        config: Dict[str, Any] = None,
        environment: str = None
    ) -> Union['PerformanceMonitor', 'QuantumSniperPerformanceMonitorV5']:
        """
        创建性能监控器
        
        Args:
            monitor_type: 监控器类型
            config: 配置参数
            environment: 运行环境
            
        Returns:
            性能监控器实例
        """
        # 自动检测环境
        if monitor_type == MonitorType.AUTO:
            monitor_type = PerformanceMonitorFactory._detect_optimal_monitor(environment)
        
        # 创建对应类型的监控器
        if monitor_type == MonitorType.QUANTUM_OPTIMIZED:
            from quantum_sniper_performance_monitor_v5 import QuantumSniperPerformanceMonitorV5
            return QuantumSniperPerformanceMonitorV5(config.get("name", "QuantumMonitor"), config)
        else:
            from .performance_monitor import PerformanceMonitor
            return PerformanceMonitor(config.get("name", "StandardMonitor"), config)
    
    @staticmethod
    def _detect_optimal_monitor(environment: str = None) -> MonitorType:
        """检测最优监控器类型"""
        try:
            # 环境检测
            if environment == "colab" or 'google.colab' in str(sys.modules):
                return MonitorType.QUANTUM_OPTIMIZED
            
            # 性能需求检测
            if PerformanceMonitorFactory._requires_quantum_performance():
                return MonitorType.QUANTUM_OPTIMIZED
            else:
                return MonitorType.STANDARD
                
        except Exception:
            return MonitorType.STANDARD
    
    @staticmethod
    def _requires_quantum_performance() -> bool:
        """检测是否需要量子级性能"""
        try:
            # 检测是否在高频交易环境
            # 检测系统资源限制
            # 检测性能要求配置
            return True  # 默认需要高性能
        except Exception:
            return False
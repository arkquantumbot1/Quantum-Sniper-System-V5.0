# src/brain/quantum_strategy_factory.py
"""
量子策略版本工厂 - 统一管理不同版本的策略
V5.0 新增 - 策略工厂模式
"""

import logging
from typing import Dict, Any, Optional, List
from core.strategy_base import BaseStrategy

# 导入版本配置
from config.strategy_versions import (
    get_recommended_version, get_version_info, 
    get_available_versions, create_strategy_instance
)

logger = logging.getLogger("strategy.factory")

class QuantumStrategyFactory:
    """量子策略工厂类 - 统一创建和管理量子策略实例"""
    
    def __init__(self):
        self.logger = logging.getLogger("strategy.factory")
        self._strategy_cache = {}  # 策略实例缓存
    
    def create_quantum_strategy(self, version: str = None, 
                              environment: str = "production",
                              config: Dict[str, Any] = None) -> Optional[BaseStrategy]:
        """创建量子神经晶格策略实例"""
        
        # 自动选择版本
        if version is None:
            version = get_recommended_version(environment)
            self.logger.info(f"自动选择版本: {version} (环境: {environment})")
        
        # 检查缓存
        cache_key = f"quantum_neural_lattice.{version}"
        if cache_key in self._strategy_cache:
            self.logger.debug(f"使用缓存的策略实例: {cache_key}")
            return self._strategy_cache[cache_key]
        
        # 创建新实例
        strategy = create_strategy_instance(
            strategy_type="quantum_neural_lattice",
            version=version, 
            config=config
        )
        
        if strategy:
            # 初始化策略
            if strategy.initialize():
                self._strategy_cache[cache_key] = strategy
                version_info = get_version_info("quantum_neural_lattice", version)
                self.logger.info(f"成功创建策略: {version_info['description']}")
                return strategy
            else:
                self.logger.error(f"策略初始化失败: {cache_key}")
                return None
        else:
            self.logger.error(f"策略创建失败: {cache_key}")
            return None
    
    def get_available_quantum_versions(self) -> Dict[str, Dict[str, Any]]:
        """获取可用的量子策略版本"""
        return get_available_versions("quantum_neural_lattice")
    
    def get_version_performance_comparison(self) -> List[Dict[str, Any]]:
        """获取版本性能对比"""
        versions = self.get_available_quantum_versions()
        comparison = []
        
        for version_id, info in versions.items():
            comparison.append({
                "version": version_id,
                "description": info["description"],
                "performance": info["performance"],
                "stability": info["stability"],
                "recommended_for": info["recommended_environments"]
            })
        
        return comparison
    
    def clear_cache(self):
        """清空策略缓存"""
        self._strategy_cache.clear()
        self.logger.info("策略缓存已清空")
    
    def get_factory_status(self) -> Dict[str, Any]:
        """获取工厂状态"""
        return {
            "cached_strategies": list(self._strategy_cache.keys()),
            "available_versions": list(self.get_available_quantum_versions().keys()),
            "total_cache_size": len(self._strategy_cache)
        }

# 创建全局工厂实例
_global_quantum_factory = QuantumStrategyFactory()

# 便捷函数
def create_quantum_strategy(version: str = None, environment: str = "production", 
                          config: Dict[str, Any] = None) -> Optional[BaseStrategy]:
    """创建量子策略的便捷函数"""
    return _global_quantum_factory.create_quantum_strategy(version, environment, config)

def get_quantum_versions() -> Dict[str, Dict[str, Any]]:
    """获取量子策略版本信息"""
    return _global_quantum_factory.get_available_quantum_versions()

def get_quantum_performance_comparison() -> List[Dict[str, Any]]:
    """获取量子策略性能对比"""
    return _global_quantum_factory.get_version_performance_comparison()

# 导出
__all__ = [
    'QuantumStrategyFactory',
    'create_quantum_strategy', 
    'get_quantum_versions',
    'get_quantum_performance_comparison',
    '_global_quantum_factory'
]

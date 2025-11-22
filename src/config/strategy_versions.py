# src/config/strategy_versions.py
"""
量子神经晶格策略版本管理配置
V5.0 新增 - 策略版本管理
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("config.strategy_versions")

# 策略版本配置
STRATEGY_VERSIONS = {
    "quantum_neural_lattice": {
        "v5_0_original": {
            "module": "brain.quantum_neural_lattice",
            "class": "QuantumNeuralLatticeStrategy",
            "description": "原始稳定版本 - 生产环境推荐",
            "performance": "5.6 signals/sec",
            "stability": "high",
            "features": "full",
            "recommended_environments": ["production", "staging"]
        },
        "v5_0_optimized": {
            "module": "brain.quantum_neural_lattice_optimized", 
            "class": "QuantumNeuralLatticeOptimized",
            "description": "性能优化版本 - 测试环境使用",
            "performance": "640 signals/sec", 
            "stability": "medium",
            "features": "optimized",
            "recommended_environments": ["testing", "development", "benchmark"]
        }
    }
}

# 环境版本映射
ENVIRONMENT_VERSION_MAPPING = {
    "production": "v5_0_original",
    "staging": "v5_0_original", 
    "testing": "v5_0_optimized",
    "development": "v5_0_optimized",
    "benchmark": "v5_0_optimized"
}

def get_recommended_version(environment: str = "production") -> str:
    """获取环境推荐版本"""
    return ENVIRONMENT_VERSION_MAPPING.get(environment, "v5_0_original")

def get_version_info(strategy_type: str, version: str) -> Optional[Dict[str, Any]]:
    """获取指定策略版本信息"""
    if strategy_type in STRATEGY_VERSIONS and version in STRATEGY_VERSIONS[strategy_type]:
        return STRATEGY_VERSIONS[strategy_type][version].copy()
    return None

def get_available_versions(strategy_type: str = "quantum_neural_lattice") -> Dict[str, Dict[str, Any]]:
    """获取可用的策略版本"""
    return STRATEGY_VERSIONS.get(strategy_type, {}).copy()

def create_strategy_instance(strategy_type: str, version: str, config: Dict[str, Any] = None):
    """创建策略实例"""
    version_info = get_version_info(strategy_type, version)
    if not version_info:
        logger.error(f"不支持的策略版本: {strategy_type}.{version}")
        return None
    
    try:
        module = __import__(version_info["module"], fromlist=[version_info["class"]])
        strategy_class = getattr(module, version_info["class"])
        return strategy_class(config=config)
    except Exception as e:
        logger.error(f"创建策略实例失败 {strategy_type}.{version}: {e}")
        return None

# 导出配置
__all__ = [
    'STRATEGY_VERSIONS',
    'ENVIRONMENT_VERSION_MAPPING', 
    'get_recommended_version',
    'get_version_info',
    'get_available_versions',
    'create_strategy_instance'
]

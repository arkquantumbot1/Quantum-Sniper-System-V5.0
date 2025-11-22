"""统一配置加载器"""
import os
import yaml
from typing import Any, Dict, Optional

# 全局配置加载器实例
_global_config_loader = None  # 在全局作用域初始化


class Config:
    """统一配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        global _global_config_loader  # 声明使用全局变量
        self.config_path = config_path
        self._config_data = {}

        # 初始化全局加载器
        if _global_config_loader is None:
            _global_config_loader = self

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        # 实现配置加载逻辑
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config_data.get(key, default)


def get_global_config() -> Config:
    """获取全局配置实例"""
    global _global_config_loader
    if _global_config_loader is None:
        _global_config_loader = Config()
    return _global_config_loader

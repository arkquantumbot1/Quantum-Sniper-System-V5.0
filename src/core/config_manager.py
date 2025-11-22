# src/core/config_manager.py
"""量子奇点系统 - 配置管理器基类 V5.0 (完全重新开发 + 极致优化)"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable
import yaml
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from threading import Lock
from dataclasses import dataclass, field
from enum import Enum

# 导入极致优化的接口定义
from interfaces import (
    IConfigManager,
    ConfigScope,
    ConfigChange,
    InterfaceMetadata,
    PerformanceMetrics,
    DataQualityLevel,
)


class ConfigFormat(Enum):
    """配置格式枚举 - 新增极致优化"""

    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    ENV = "env"
    PYTHON = "python"


class ConfigValidationLevel(Enum):
    """配置验证级别 - 新增极致优化"""

    STRICT = "strict"  # 严格验证，任何错误都失败
    RELAXED = "relaxed"  # 宽松验证，只记录警告
    NONE = "none"  # 不验证


@dataclass
class ConfigMetadata:
    """配置元数据 - 新增极致优化"""

    version: str = "1.0.0"
    description: str = ""
    author: str = "Quantum-Sniper-Team"
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    format: ConfigFormat = ConfigFormat.YAML
    validation_level: ConfigValidationLevel = ConfigValidationLevel.STRICT


class BaseConfigManager(IConfigManager):
    """配置管理器基类 V5.0 - 完全重新开发 + 极致优化"""

    # 接口元数据 - 新增极致优化
    _metadata = InterfaceMetadata(
        version="5.0",
        description="统一配置管理器接口",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={"config_load_time": 0.01, "config_validation_time": 0.005},
        dependencies=["IEventDispatcher", "IDataProcessor"],
        compatibility=["4.2", "4.1"],
    )

    def __init__(
        self, config_path: str = None, scope: ConfigScope = ConfigScope.GLOBAL
    ):
        self.config_path = config_path
        self.scope = scope
        self.config: Dict[str, Any] = {}
        self.config_history: List[ConfigChange] = []
        self.config_watchers: Dict[str, List[Callable]] = {}

        # 性能监控 - 新增极致优化
        self._performance_metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0,
            cpu_usage=0.0,
            call_count=0,
            error_count=0,
            cache_hit_rate=0.0,
        )

        # 智能缓存 - 新增极致优化
        self._config_cache: Dict[str, Any] = {}
        self._schema_cache: Dict[str, Any] = {}
        self._checksum_cache: str = ""

        # 线程安全 - 新增极致优化
        self._lock = Lock()
        self._watcher_lock = Lock()

        # 配置元数据 - 新增极致优化
        self._metadata_info = ConfigMetadata()

        self.logger = logging.getLogger(f"config.{scope.value}")

        # 自动初始化
        self._auto_initialize()

    @classmethod
    def get_interface_metadata(cls) -> InterfaceMetadata:
        """获取接口元数据 - 新增极致优化"""
        return cls._metadata

    def _auto_initialize(self):
        """自动初始化 - 新增极致优化"""
        try:
            if self.config_path and Path(self.config_path).exists():
                self.load_config()
                self._update_metadata()
        except Exception as e:
            self.logger.warning(f"自动初始化失败: {e}")

    def _update_metadata(self):
        """更新配置元数据 - 新增极致优化"""
        self._metadata_info.last_modified = datetime.now()
        if self.config_path:
            config_content = str(self.config)
            self._metadata_info.checksum = hashlib.md5(
                config_content.encode()
            ).hexdigest()

    @abstractmethod
    def load_config(self) -> bool:
        """加载配置 - 极致优化版本"""
        pass

    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值 - 极致优化版本"""
        pass

    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """设置配置值 - 极致优化版本"""
        pass

    @abstractmethod
    def validate_config(self) -> Tuple[bool, List[str]]:
        """验证配置完整性 - 极致优化版本"""
        pass

    # 🚀 新增极致优化方法

    def hot_reload_config(self) -> bool:
        """热重载配置 - 新增极致优化"""
        start_time = datetime.now()

        try:
            with self._lock:
                old_config = self.config.copy()
                old_checksum = self._metadata_info.checksum

                # 重新加载配置
                success = self.load_config()
                if not success:
                    return False

                # 检查配置是否实际变化
                new_checksum = self._metadata_info.checksum
                if new_checksum == old_checksum:
                    self.logger.info("配置未变化，跳过热重载")
                    return True

                # 记录配置变更
                changes = self._detect_config_changes(old_config, self.config)
                for change in changes:
                    self.config_history.append(change)
                    self._notify_watchers(change)

                # 更新性能指标
                reload_time = (datetime.now() - start_time).total_seconds()
                self._performance_metrics.execution_time += reload_time
                self._performance_metrics.call_count += 1

                self.logger.info(
                    f"配置热重载完成，检测到 {len(changes)} 处变更，耗时: {reload_time:.3f}s"
                )
                return True

        except Exception as e:
            self.logger.error(f"配置热重载失败: {e}")
            self._performance_metrics.error_count += 1
            return False

    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式 - 新增极致优化"""
        schema_key = f"schema_{self.scope.value}"

        # 使用缓存提高性能
        if schema_key in self._schema_cache:
            self._performance_metrics.cache_hit_rate += 1
            return self._schema_cache[schema_key]

        try:
            schema = {
                "scope": self.scope.value,
                "version": self._metadata_info.version,
                "description": self._metadata_info.description,
                "structure": self._generate_config_structure(),
                "required_fields": self._get_required_fields(),
                "validation_rules": self._get_validation_rules(),
                "metadata": {
                    "last_modified": self._metadata_info.last_modified.isoformat(),
                    "checksum": self._metadata_info.checksum,
                    "format": self._metadata_info.format.value,
                },
            }

            # 缓存schema
            self._schema_cache[schema_key] = schema
            return schema

        except Exception as e:
            self.logger.error(f"生成配置模式失败: {e}")
            return {}

    def watch_config(self, key: str, callback: Callable[[ConfigChange], None]) -> bool:
        """监控配置变更 - 新增极致优化"""
        try:
            with self._watcher_lock:
                if key not in self.config_watchers:
                    self.config_watchers[key] = []

                if callback not in self.config_watchers[key]:
                    self.config_watchers[key].append(callback)
                    self.logger.debug(f"已注册配置监控: {key} -> {callback.__name__}")
                    return True
                else:
                    self.logger.warning(f"回调函数已注册: {key} -> {callback.__name__}")
                    return False

        except Exception as e:
            self.logger.error(f"注册配置监控失败: {e}")
            return False

    def get_config_history(self, key: str) -> List[ConfigChange]:
        """获取配置历史 - 新增极致优化"""
        if not key:
            return self.config_history[-10:]  # 返回最近10条变更

        return [change for change in self.config_history if change.key == key][-10:]

    def rollback_config(self, key: str, steps: int = 1) -> bool:
        """回滚配置 - 新增极致优化"""
        try:
            with self._lock:
                # 获取相关历史记录
                relevant_history = [h for h in self.config_history if h.key == key]
                if not relevant_history or len(relevant_history) < steps:
                    self.logger.warning(f"无法回滚配置 {key}，历史记录不足")
                    return False

                # 执行回滚
                target_change = relevant_history[-steps]
                old_value = target_change.old_value

                # 设置回滚值
                success = self.set_config(key, old_value)
                if success:
                    rollback_change = ConfigChange(
                        key=key,
                        old_value=self.config.get(key),
                        new_value=old_value,
                        timestamp=datetime.now(),
                        source="rollback",
                        reason=f"回滚到 {target_change.timestamp} 的状态",
                    )
                    self.config_history.append(rollback_change)
                    self.logger.info(f"配置回滚成功: {key} -> 步骤 {steps}")

                return success

        except Exception as e:
            self.logger.error(f"配置回滚失败: {e}")
            self._performance_metrics.error_count += 1
            return False

    def optimize_config_storage(self) -> bool:
        """优化配置存储 - 新增极致优化"""
        try:
            # 清理过期的历史记录（保留最近100条）
            if len(self.config_history) > 100:
                self.config_history = self.config_history[-100:]

            # 清理缓存
            self._config_cache.clear()

            # 压缩配置数据
            self._compress_config_data()

            # 更新性能指标
            self._performance_metrics.memory_usage = self._calculate_memory_usage()

            self.logger.info("配置存储优化完成")
            return True

        except Exception as e:
            self.logger.error(f"配置存储优化失败: {e}")
            return False

    def get_config_by_scope(self, scope: ConfigScope) -> Dict[str, Any]:
        """按作用域获取配置 - 新增极致优化"""
        if scope == self.scope:
            return self.config.copy()

        # 对于不同的作用域，返回空配置
        # 实际实现中可能需要从其他配置管理器获取
        return {}

    async def load_config_async(self) -> bool:
        """异步加载配置 - 新增极致优化"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.load_config
            )
        except Exception as e:
            self.logger.error(f"异步加载配置失败: {e}")
            return False

    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标 - 新增极致优化"""
        return self._performance_metrics

    def validate_config_advanced(self) -> Tuple[bool, Dict[str, Any]]:
        """高级配置验证 - 新增极致优化"""
        validation_result = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "quality_score": 0.0,
        }

        try:
            # 基础验证
            is_valid, errors = self.validate_config()
            validation_result["is_valid"] = is_valid
            validation_result["errors"] = errors

            # 高级验证
            if is_valid:
                validation_result.update(self._perform_advanced_validation())

            # 计算质量分数
            validation_result["quality_score"] = self._calculate_quality_score(
                validation_result
            )

            return is_valid, validation_result

        except Exception as e:
            self.logger.error(f"高级配置验证失败: {e}")
            validation_result["errors"].append(f"验证过程异常: {e}")
            return False, validation_result

    # 🔧 内部辅助方法

    def _detect_config_changes(
        self, old_config: Dict, new_config: Dict
    ) -> List[ConfigChange]:
        """检测配置变更 - 内部方法"""
        changes = []
        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)

            if old_value != new_value:
                change = ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=datetime.now(),
                    source="hot_reload",
                    reason="配置热重载检测到变更",
                )
                changes.append(change)

        return changes

    def _notify_watchers(self, change: ConfigChange):
        """通知监控器 - 内部方法"""
        if change.key in self.config_watchers:
            for callback in self.config_watchers[change.key]:
                try:
                    callback(change)
                except Exception as e:
                    self.logger.error(f"配置监控回调执行失败: {e}")

    def _generate_config_structure(self) -> Dict[str, Any]:
        """生成配置结构 - 内部方法"""

        def analyze_structure(data, path=""):
            structure = {}
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    structure[key] = {
                        "type": type(value).__name__,
                        "path": current_path,
                        "children": analyze_structure(value, current_path)
                        if isinstance(value, dict)
                        else None,
                    }
            return structure

        return analyze_structure(self.config)

    def _get_required_fields(self) -> List[str]:
        """获取必需字段 - 内部方法"""
        # 基于配置模式返回必需字段
        # 这里可以根据具体实现返回不同的必需字段列表
        required_fields = []

        if self.scope == ConfigScope.STRATEGY:
            required_fields = ["name", "enabled", "risk_level"]
        elif self.scope == ConfigScope.RISK:
            required_fields = ["max_drawdown", "position_sizing", "stop_loss"]

        return required_fields

    def _get_validation_rules(self) -> Dict[str, Any]:
        """获取验证规则 - 内部方法"""
        # 基于配置作用域返回验证规则
        rules = {}

        if self.scope == ConfigScope.STRATEGY:
            rules = {
                "risk_level": {"type": "string", "allowed": ["low", "medium", "high"]},
                "enabled": {"type": "boolean"},
            }
        elif self.scope == ConfigScope.RISK:
            rules = {
                "max_drawdown": {"type": "number", "min": 0, "max": 100},
                "position_sizing": {"type": "number", "min": 0, "max": 1},
            }

        return rules

    def _compress_config_data(self):
        """压缩配置数据 - 内部方法"""
        # 移除空值和None值
        self.config = {k: v for k, v in self.config.items() if v is not None}

    def _calculate_memory_usage(self) -> int:
        """计算内存使用 - 内部方法"""
        import sys

        return sys.getsizeof(self.config) + sys.getsizeof(self.config_history)

    def _perform_advanced_validation(self) -> Dict[str, Any]:
        """执行高级验证 - 内部方法"""
        result = {"warnings": [], "suggestions": []}

        # 检查配置合理性
        if self.scope == ConfigScope.RISK:
            max_drawdown = self.config.get("max_drawdown")
            if max_drawdown and max_drawdown > 50:
                result["warnings"].append("最大回撤设置过高，建议调整到20%以下")

        # 检查配置一致性
        if "timeframe" in self.config and "interval" in self.config:
            if self.config["timeframe"] == "1m" and self.config["interval"] > 300:
                result["suggestions"].append("1分钟时间框架建议使用较小的间隔")

        return result

    def _calculate_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """计算质量分数 - 内部方法"""
        base_score = 100.0

        # 根据错误数量扣分
        error_penalty = len(validation_result["errors"]) * 20
        warning_penalty = len(validation_result["warnings"]) * 5

        final_score = max(0, base_score - error_penalty - warning_penalty)
        return final_score / 100.0  # 归一化到0-1

    def __str__(self) -> str:
        return f"ConfigManager(scope={self.scope.value}, items={len(self.config)})"

    def __repr__(self) -> str:
        return (
            f"BaseConfigManager(scope={self.scope.value}, "
            f"path={self.config_path}, config_items={len(self.config)})"
        )


# 配置管理器工厂 - 新增极致优化
class ConfigManagerFactory:
    """配置管理器工厂 - 支持动态创建和管理"""

    _managers: Dict[ConfigScope, BaseConfigManager] = {}

    @classmethod
    def create_manager(
        cls, scope: ConfigScope, config_path: str = None
    ) -> BaseConfigManager:
        """创建配置管理器"""
        from interfaces import InterfaceRegistry

        # 查找已注册的配置管理器实现
        manager_class = None
        for interface_name in InterfaceRegistry.list_interfaces():
            interface_class = InterfaceRegistry.get_interface(interface_name)
            if (
                interface_class
                and hasattr(interface_class, "_metadata")
                and "config" in interface_class._metadata.description.lower()
                and issubclass(interface_class, BaseConfigManager)
            ):
                manager_class = interface_class
                break

        if not manager_class:
            # 如果没有找到特定实现，使用基础类
            manager_class = BaseConfigManager

        manager = manager_class(config_path, scope)
        cls._managers[scope] = manager
        return manager

    @classmethod
    def get_manager(cls, scope: ConfigScope) -> Optional[BaseConfigManager]:
        """获取配置管理器"""
        return cls._managers.get(scope)

    @classmethod
    def reload_all_managers(cls) -> bool:
        """重新加载所有管理器"""
        success = True
        for manager in cls._managers.values():
            if not manager.hot_reload_config():
                success = False
        return success


# 自动注册接口
from interfaces import InterfaceRegistry

InterfaceRegistry.register_interface(BaseConfigManager)

__all__ = [
    "BaseConfigManager",
    "ConfigManagerFactory",
    "ConfigFormat",
    "ConfigValidationLevel",
    "ConfigMetadata",
]

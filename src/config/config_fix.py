# src/config/config_fix.py
"""
量子奇点狙击系统 - 配置系统紧急修复补丁 V5.0
修复配置加载和验证的核心问题
"""

import os
import yaml
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum
from pathlib import Path
import logging


class ConfigScope(Enum):
    GLOBAL = "global"
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"


class ConfigFixManager:
    """配置系统修复管理器"""

    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getenv(
            "QUANTUM_SNIPER_ROOT",
            "C:/Users/User/TradingProjects/Quantum-Sniper-System/Quantum-Sniper-System-V5.0",
        )
        self.src_path = os.path.join(self.project_root, "src")
        self.config_path = os.path.join(self.src_path, "config.yaml")
        self.logger = logging.getLogger("config_fix")

        # 确保必要的目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.project_root,
            self.src_path,
            os.path.join(self.project_root, "data"),
            os.path.join(self.project_root, "logs"),
            os.path.join(self.project_root, "tests"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def create_fixed_config(self) -> bool:
        """创建修复版配置文件"""
        try:
            fixed_config = self._get_fixed_config_content()

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(fixed_config, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"修复版配置文件已创建: {self.config_path}")
            return True

        except Exception as e:
            self.logger.error(f"创建修复版配置文件失败: {e}")
            return False

    def _get_fixed_config_content(self) -> Dict[str, Any]:
        """获取修复版配置内容"""
        return {
            "system": {
                "name": "Quantum-Sniper-System-V5.0",
                "version": "5.0.0",
                "environment": "development",
                "log_level": "INFO",
            },
            "order_executor": {
                "execution_mode": "simulation",
                "enabled_exchanges": [
                    "binance",
                    "bybit",
                    "okx",
                    "bingx",
                    "bitget",
                    "mexc",
                ],
                "default_exchange": "binance",
                "fpga_enabled": False,
                "large_order_threshold": 10000.0,
                "retry_config": {
                    "max_retries": 3,
                    "retry_delay": 0.1,
                    "backoff_factor": 2.0,
                },
            },
            "risk_management": {
                "enabled": True,
                "max_position_size": 10000.0,
                "max_drawdown": 0.1,
                "volatility_threshold": 0.15,
                "concentration_limit": 0.8,
                "stop_loss_enabled": True,
                "take_profit_enabled": True,
            },
            "trading": {
                "enabled": True,
                "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                "default_quantity": 0.01,
                "max_quantity": 10.0,
                "min_quantity": 0.001,
                "leverage": 1,
            },
            "performance": {
                "cache_enabled": True,
                "cache_size": 1000,
                "parallel_processing": True,
                "max_workers": 10,
            },
            "sac_strategy_optimizer": {
                "name": "QuantumSACOptimizer",
                "enabled": True,
                "risk_level": "medium",
                "population_size": 100,
                "mutation_rate": 0.15,
                "crossover_rate": 0.85,
                "max_generations": 2000,
                "elite_count": 10,
                "quantum_enhancement": True,
                "quantum_coherence_level": 0.8,
                "quantum_entanglement": True,
                "learning_rate": 0.001,
                "batch_size": 64,
                "replay_buffer_size": 100000,
                "target_update_interval": 100,
                "early_stopping": True,
                "performance_threshold": 0.8,
                "convergence_tolerance": 0.01,
                "use_gpu": True,
                "max_memory_usage": 0.8,
                "parallel_training": True,
            },
            "market_data": {
                "update_interval": 1.0,
                "history_days": 30,
                "real_time_enabled": True,
                "cache_size": 10000,
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 60,
                "health_check_interval": 30,
                "alert_enabled": True,
            },
        }

    def validate_config_file(self) -> Tuple[bool, List[str]]:
        """验证配置文件"""
        errors = []

        try:
            # 检查文件是否存在
            if not os.path.exists(self.config_path):
                errors.append(f"配置文件不存在: {self.config_path}")
                return False, errors

            # 检查文件可读性
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_content = yaml.safe_load(f)

            if not config_content:
                errors.append("配置文件为空或格式错误")
                return False, errors

            # 验证必要配置段
            required_sections = [
                "system",
                "order_executor",
                "risk_management",
                "trading",
            ]
            for section in required_sections:
                if section not in config_content:
                    errors.append(f"缺少必要配置段: {section}")

            # 验证订单执行器配置
            order_executor_config = config_content.get("order_executor", {})
            required_order_fields = [
                "execution_mode",
                "enabled_exchanges",
                "default_exchange",
            ]
            for field in required_order_fields:
                if field not in order_executor_config:
                    errors.append(f"订单执行器缺少配置字段: {field}")

            return len(errors) == 0, errors

        except Exception as e:
            errors.append(f"配置文件验证异常: {e}")
            return False, errors

    def backup_original_config(self) -> bool:
        """备份原始配置文件"""
        try:
            if os.path.exists(self.config_path):
                backup_path = self.config_path + ".backup"
                import shutil

                shutil.copy2(self.config_path, backup_path)
                self.logger.info(f"原始配置文件已备份: {backup_path}")
                return True
            return True  # 如果没有原始文件，也返回成功
        except Exception as e:
            self.logger.error(f"备份配置文件失败: {e}")
            return False


class PatchedOrderExecutor:
    """修补后的订单执行器 - 临时解决方案"""

    def __init__(self, config_path: str = None):
        self.config_fix = ConfigFixManager()
        self.config_path = config_path or self.config_fix.config_path
        self.config = None
        self.logger = logging.getLogger("patched_order_executor")

        # 初始化配置
        self._initialize_config()

    def _initialize_config(self):
        """初始化配置"""
        try:
            # 确保配置文件存在且有效
            if not os.path.exists(self.config_path):
                self.logger.warning("配置文件不存在，创建修复版配置")
                self.config_fix.create_fixed_config()

            # 验证配置文件
            valid, errors = self.config_fix.validate_config_file()
            if not valid:
                self.logger.warning(f"配置文件验证失败: {errors}")
                self.config_fix.create_fixed_config()

            # 加载配置
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            self.logger.info("配置加载成功")

        except Exception as e:
            self.logger.error(f"配置初始化失败: {e}")
            # 使用默认配置作为降级方案
            self.config = self.config_fix._get_fixed_config_content()

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if not self.config:
            return default

        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def validate_config(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        return self.config_fix.validate_config_file()

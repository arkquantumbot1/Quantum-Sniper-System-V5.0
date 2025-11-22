# src/brain/quantum_neural_lattice_optimized.py
"""
量子神经晶格优化版本 - 修复抽象方法问题
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import math

# 保持现有版本的导入兼容性
from interfaces import (
    IStrategySignal, SignalDirection, SignalPriority, PerformanceMetrics,
    InterfaceMetadata, SignalMetadata, DataQualityLevel, MarketRegime
)
from core.strategy_base import BaseStrategy, StrategySignal, StrategyError

class QuantumNeuralLatticeOptimized(BaseStrategy):
    """量子神经晶格策略优化版 - 修复抽象方法版本"""
    
    def __init__(self, name: str = "QuantumNeuralLatticeOptimized", config: Dict[str, Any] = None):
        if config is None:
            config = {}
            
        # 性能优化配置
        optimized_config = {
            **config,
            "uncertainty_samples": config.get("uncertainty_samples", 10),  # 减少采样次数
            "enable_batch_inference": config.get("enable_batch_inference", True),
            "use_gpu_if_available": config.get("use_gpu_if_available", True),
            "cache_predictions": config.get("cache_predictions", True),
            "use_fast_mode": config.get("use_fast_mode", True)  # 默认使用快速模式
        }
        
        super().__init__(name, optimized_config)
        
        # 性能优化参数
        self.uncertainty_samples = self.config.get("uncertainty_samples", 10)
        self.enable_batch_inference = self.config.get("enable_batch_inference", True)
        self.use_gpu = self.config.get("use_gpu_if_available", True) and torch.cuda.is_available()
        self.cache_predictions = self.config.get("cache_predictions", True)
        self.use_fast_mode = self.config.get("use_fast_mode", True)
        
        # 预测缓存
        self._prediction_cache = {}
        self._cache_max_size = 100
        
        # 模型初始化（延迟加载）
        self.model = None
        self.model_initialized = False
        
        self.logger = logging.getLogger(f"strategy.quantum_lattice_optimized")
        
    def initialize(self) -> bool:
        """优化版初始化"""
        try:
            self.logger.info("开始初始化优化版量子神经晶格策略...")
            
            # 延迟加载模型以减少启动时间
            from .quantum_neural_lattice import QuantumNeuralLatticeModel
            
            self.model = QuantumNeuralLatticeModel(
                input_dim=self.config.get("input_dim", 64),
                hidden_dims=self.config.get("hidden_dims", [128, 256, 128, 64]),
                output_dim=self.config.get("output_dim", 4),
                quantum_entanglement=self.config.get("quantum_entanglement", 0.7)
            )
            
            # GPU加速
            if self.use_gpu:
                self.model = self.model.cuda()
                self.logger.info("✅ 启用GPU加速")
            
            self.model_initialized = True
            self.initialized = True
            
            self.logger.info("优化版量子神经晶格策略初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"优化版量子神经晶格策略初始化失败: {e}")
            return False

    # 🔧 修复：实现基类要求的抽象方法
    def get_signal(self, data: Any) -> Optional[IStrategySignal]:
        """实现基类抽象方法 - 根据配置选择模式"""
        if self.use_fast_mode:
            return self.get_signal_fast(data)
        else:
            return self.get_signal_with_optimized_uncertainty(data)
    
    def get_status(self) -> Dict[str, Any]:
        """实现基类抽象方法 - 获取策略状态"""
        base_status = super().get_status()
        
        optimized_status = {
            **base_status,
            "model_initialized": self.model_initialized,
            "use_fast_mode": self.use_fast_mode,
            "uncertainty_samples": self.uncertainty_samples,
            "use_gpu": self.use_gpu,
            "cache_predictions": self.cache_predictions,
            "performance_metrics": self._performance_metrics.to_dict(),
            "cache_size": len(self._prediction_cache)
        }
        
        return optimized_status
    
    def get_signal_fast(self, data: Any) -> Optional[IStrategySignal]:
        """快速信号生成 - 跳过不确定性计算"""
        if not self.model_initialized or self.model is None:
            return None
        
        try:
            start_time = datetime.now()
            
            # 数据预处理
            processed_data = self._preprocess_data(data)
            if processed_data is None:
                return None
            
            # 转换为张量
            input_tensor = torch.FloatTensor(processed_data).unsqueeze(0)
            if self.use_gpu:
                input_tensor = input_tensor.cuda()
            
            # 快速预测 - 不使用不确定性
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction = prediction.squeeze(0).cpu()
            
            # 解析预测结果
            signal_strength = float(prediction[0].item())
            direction_confidence = float(prediction[1].item())
            
            # 确定信号方向
            direction = self._determine_signal_direction_fast(direction_confidence)
            
            # 计算置信度（简化版）
            final_confidence = max(0.0, min(1.0, abs(direction_confidence)))
            
            # 置信度阈值检查
            if final_confidence < 0.6:
                return None
            
            # 创建信号
            signal_metadata = SignalMetadata(
                source="quantum_neural_lattice_optimized",
                priority=SignalPriority.HIGH if final_confidence > 0.8 else SignalPriority.MEDIUM,
                tags=["quantum", "optimized", "fast"],
                confidence_interval=(final_confidence, final_confidence)
            )
            
            signal_data = {
                "signal_strength": signal_strength,
                "direction_confidence": direction_confidence,
                "confidence": final_confidence,
                "inference_time": (datetime.now() - start_time).total_seconds(),
                "optimized": True,
                "mode": "fast"
            }
            
            signal = StrategySignal(
                signal_type="QUANTUM_LATTICE_OPTIMIZED",
                confidence=final_confidence,
                data=signal_data,
                direction=direction,
                metadata=signal_metadata
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"快速信号生成失败: {e}")
            return None
    
    def get_signal_with_optimized_uncertainty(self, data: Any) -> Optional[IStrategySignal]:
        """带优化不确定性计算的信号生成"""
        if not self.model_initialized or self.model is None:
            return None
        
        try:
            start_time = datetime.now()
            
            # 数据预处理
            processed_data = self._preprocess_data(data)
            if processed_data is None:
                return None
            
            # 转换为张量
            input_tensor = torch.FloatTensor(processed_data).unsqueeze(0)
            if self.use_gpu:
                input_tensor = input_tensor.cuda()
            
            # 优化版不确定性预测 - 减少采样次数
            with torch.no_grad():
                predictions = []
                
                for i in range(self.uncertainty_samples):
                    # 添加少量噪声
                    quantum_noise = torch.randn_like(input_tensor) * 0.01
                    noisy_input = input_tensor + quantum_noise
                    
                    prediction = self.model(noisy_input)
                    predictions.append(prediction)
                
                predictions_tensor = torch.stack(predictions)
                mean_prediction = torch.mean(predictions_tensor, dim=0).squeeze(0).cpu()
                uncertainty = torch.std(predictions_tensor, dim=0).squeeze(0).cpu()
            
            # 解析预测结果
            signal_strength = float(mean_prediction[0].item())
            direction_confidence = float(mean_prediction[1].item())
            total_uncertainty = float(torch.mean(uncertainty).item())
            
            # 确定信号方向
            direction = self._determine_signal_direction(direction_confidence, total_uncertainty)
            
            # 计算最终置信度
            final_confidence = max(0.0, min(1.0, 
                abs(direction_confidence) * (1 - total_uncertainty)
            ))
            
            # 置信度阈值检查
            if final_confidence < 0.6 or total_uncertainty > self.config.get("uncertainty_threshold", 0.3):
                return None
            
            # 创建信号
            signal_metadata = SignalMetadata(
                source="quantum_neural_lattice_optimized",
                priority=SignalPriority.HIGH if final_confidence > 0.8 else SignalPriority.MEDIUM,
                tags=["quantum", "optimized", "uncertainty"],
                confidence_interval=(final_confidence - total_uncertainty, final_confidence + total_uncertainty)
            )
            
            signal_data = {
                "signal_strength": signal_strength,
                "direction_confidence": direction_confidence,
                "uncertainty": total_uncertainty,
                "confidence": final_confidence,
                "inference_time": (datetime.now() - start_time).total_seconds(),
                "samples_used": self.uncertainty_samples,
                "optimized": True,
                "mode": "uncertainty"
            }
            
            signal = StrategySignal(
                signal_type="QUANTUM_LATTICE_OPTIMIZED",
                confidence=final_confidence,
                data=signal_data,
                direction=direction,
                metadata=signal_metadata
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"优化不确定性信号生成失败: {e}")
            return None
    
    def _determine_signal_direction_fast(self, direction_confidence: float) -> SignalDirection:
        """快速信号方向判断"""
        if direction_confidence > 0.1:
            return SignalDirection.LONG
        elif direction_confidence < -0.1:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL
    
    def _determine_signal_direction(self, direction_confidence: float, uncertainty: float) -> SignalDirection:
        """带不确定性的信号方向判断"""
        adjusted_confidence = direction_confidence * (1 - uncertainty)
        
        if adjusted_confidence > 0.1:
            return SignalDirection.LONG
        elif adjusted_confidence < -0.1:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL
    
    def _preprocess_data(self, data: Any) -> Optional[np.ndarray]:
        """数据预处理 - 与原始版本保持一致"""
        try:
            if isinstance(data, dict):
                features = []
                
                # 价格相关特征
                price_features = [
                    data.get('open', 50000.0),
                    data.get('high', 51000.0), 
                    data.get('low', 49000.0),
                    data.get('close', 50500.0),
                    data.get('volume', 1000000)
                ]
                features.extend(price_features)
                
                # 技术指标特征
                tech_indicators = data.get('technical_indicators', {})
                features.extend([
                    tech_indicators.get('rsi', 50.0),
                    tech_indicators.get('macd', 0.0),
                    tech_indicators.get('bollinger_upper', 52000.0),
                    tech_indicators.get('bollinger_lower', 48000.0),
                    tech_indicators.get('atr', 500.0)
                ])
                
                # 填充到固定维度
                target_dim = self.config.get("input_dim", 64)
                if len(features) < target_dim:
                    features.extend([0.0] * (target_dim - len(features)))
                elif len(features) > target_dim:
                    features = features[:target_dim]
                
                return np.array(features, dtype=np.float32)
                
            elif isinstance(data, (list, np.ndarray)):
                processed = np.array(data, dtype=np.float32).flatten()
                target_dim = self.config.get("input_dim", 64)
                
                if len(processed) < target_dim:
                    processed = np.pad(processed, (0, target_dim - len(processed)))
                elif len(processed) > target_dim:
                    processed = processed[:target_dim]
                
                return processed
                
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            return None

# 导出优化版本
__all__ = ['QuantumNeuralLatticeOptimized']

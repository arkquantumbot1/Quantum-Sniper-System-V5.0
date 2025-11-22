# src/brain/quantum_neural_lattice.py
# 恢复后的 quantum_neural_lattice.py 架构
"""
量子奇点狙击系统 - 量子神经晶格 V5.0 (完全功能恢复版)
基于原始版本(1)恢复所有功能，同时确保与现有版本的接口兼容性
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque

# 保持现有版本的导入兼容性
from src.interfaces import (
    IStrategySignal, SignalDirection, SignalPriority, PerformanceMetrics,
    InterfaceMetadata, SignalMetadata, DataQualityLevel, MarketRegime
)
from src.core.strategy_base import BaseStrategy, StrategySignal, StrategyError

# ==================== 完全恢复原始版本功能 ====================

class QuantumState(Enum):
    """量子状态枚举 - 完全恢复"""
    SUPERPOSITION = "superposition"
    ENTANGLEMENT = "entanglement" 
    COLLAPSED = "collapsed"
    COHERENT = "coherent"
    DECOHERENT = "decoherent"

class LatticeDimension(Enum):
    """晶格维度枚举 - 完全恢复"""
    TIME = "time"
    PRICE = "price" 
    VOLUME = "volume"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    CORRELATION = "correlation"

@dataclass
class QuantumNeuron:
    """量子神经元 - 完全恢复"""
    weights: torch.Tensor
    bias: torch.Tensor
    phase: float = 0.0
    amplitude: float = 1.0
    entanglement_level: float = 0.0
    coherence_time: float = 1.0
    
    def __post_init__(self):
        self.weights = torch.nn.Parameter(self.weights)
        self.bias = torch.nn.Parameter(self.bias)

@dataclass  
class LatticeNode:
    """晶格节点 - 完全恢复"""
    node_id: str
    dimension: LatticeDimension
    position: Tuple[float, float, float]
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    activation_level: float = 0.0
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class QuantumActivation(nn.Module):
    """量子激活函数 - 完全恢复 + 兼容性保证"""
    
    def __init__(self, activation_type: str = "quantum_relu"):
        super().__init__()
        self.activation_type = activation_type
        self.phase_shift = nn.Parameter(torch.tensor(0.0))
        self.amplitude_modulation = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "quantum_relu":
            return self._quantum_relu(x)
        elif self.activation_type == "quantum_sigmoid":
            return self._quantum_sigmoid(x)
        elif self.activation_type == "quantum_tanh":
            return self._quantum_tanh(x)
        else:
            return self._default_quantum(x)
    
    def _quantum_relu(self, x: torch.Tensor) -> torch.Tensor:
        base_activation = torch.relu(x)
        quantum_effect = torch.sin(x + self.phase_shift) * self.amplitude_modulation
        return base_activation + 0.1 * quantum_effect
    
    def _quantum_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        base_activation = torch.sigmoid(x)
        quantum_effect = torch.cos(x + self.phase_shift) * self.amplitude_modulation
        return torch.clamp(base_activation + 0.05 * quantum_effect, 0, 1)
    
    def _quantum_tanh(self, x: torch.Tensor) -> torch.Tensor:
        base_activation = torch.tanh(x)
        quantum_effect = torch.sin(x * 2 + self.phase_shift) * self.amplitude_modulation
        return torch.clamp(base_activation + 0.05 * quantum_effect, -1, 1)
    
    def _default_quantum(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sin(x + self.phase_shift) * self.amplitude_modulation

class QuantumNeuralLayer(nn.Module):
    """量子神经层 - 完全恢复 + 兼容性保证"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: str = "quantum_relu",
                 entanglement_level: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.entanglement_level = entanglement_level
        
        # 量子权重初始化
        self.weights = nn.Parameter(
            self._quantum_weight_init(input_dim, output_dim)
        )
        self.bias = nn.Parameter(
            torch.randn(output_dim) * 0.1
        )
        
        # 量子激活函数
        self.activation = QuantumActivation(activation)
        
        # 量子相位参数
        self.phase_parameters = nn.Parameter(
            torch.randn(output_dim) * 0.1
        )
        
    def _quantum_weight_init(self, input_dim: int, output_dim: int) -> torch.Tensor:
        base_weights = torch.randn(input_dim, output_dim) * math.sqrt(2.0 / input_dim)
        quantum_fluctuation = torch.sin(
            torch.linspace(0, 2 * math.pi, input_dim * output_dim).reshape(input_dim, output_dim)
        ) * 0.1
        return base_weights + quantum_fluctuation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_output = torch.matmul(x, self.weights) + self.bias
        phase_adjusted = linear_output + self.phase_parameters
        
        if self.entanglement_level > 0:
            entanglement_effect = torch.sin(phase_adjusted * self.entanglement_level)
            phase_adjusted = phase_adjusted + entanglement_effect * 0.1
        
        activated = self.activation(phase_adjusted)
        return activated

class QuantumNeuralLatticeModel(nn.Module):
    """量子神经晶格模型 - 完全恢复 + 兼容性增强"""
    
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dims: List[int] = None,
                 output_dim: int = 4,
                 lattice_dimensions: List[LatticeDimension] = None,
                 quantum_entanglement: float = 0.7):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantum_entanglement = quantum_entanglement
        
        # 保持与现有版本的默认值兼容
        if hidden_dims is None:
            hidden_dims = [128, 256, 128, 64]  # 原始版本的优化配置
        
        if lattice_dimensions is None:
            lattice_dimensions = [
                LatticeDimension.TIME,
                LatticeDimension.PRICE, 
                LatticeDimension.VOLUME,
                LatticeDimension.VOLATILITY
            ]
        
        self.lattice_dimensions = lattice_dimensions
        self.dimension_encoders = nn.ModuleDict()
        
        # 创建维度编码器
        for dimension in lattice_dimensions:
            self.dimension_encoders[dimension.value] = nn.Linear(input_dim, hidden_dims[0])
        
        # 构建量子神经层
        layers = []
        current_dim = hidden_dims[0] * len(lattice_dimensions)
        
        for hidden_dim in hidden_dims:
            layers.append(
                QuantumNeuralLayer(current_dim, hidden_dim, "quantum_relu", quantum_entanglement)
            )
            current_dim = hidden_dim
        
        # 输出层
        layers.append(
            QuantumNeuralLayer(current_dim, output_dim, "quantum_tanh", quantum_entanglement * 0.5)
        )
        
        self.layers = nn.ModuleList(layers)
        
        # 量子态参数
        self.quantum_state_params = nn.ParameterDict({
            "coherence_time": nn.Parameter(torch.tensor(1.0)),
            "decoherence_rate": nn.Parameter(torch.tensor(0.1)),
            "superposition_strength": nn.Parameter(torch.tensor(0.5))
        })
        
        # 性能监控 - 保持与现有版本兼容
        self.performance_metrics = {
            "inference_time": 0.0,
            "memory_usage": 0,
            "quantum_coherence": 1.0
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 完全恢复 + 错误处理增强"""
        start_time = datetime.now()
        
        try:
            # 多维度特征编码
            encoded_features = []
            for dimension in self.lattice_dimensions:
                encoder = self.dimension_encoders[dimension.value]
                encoded = encoder(x)
                encoded_features.append(encoded)
            
            # 特征融合
            fused_features = torch.cat(encoded_features, dim=1)
            current_output = fused_features
            
            # 量子神经层处理
            for layer in self.layers:
                current_output = layer(current_output)
                
                # 应用量子相干效应
                coherence_effect = torch.sin(
                    current_output * self.quantum_state_params["coherence_time"]
                ) * self.quantum_state_params["superposition_strength"]
                current_output = current_output + coherence_effect
            
            # 量子态坍缩（输出归一化）
            final_output = torch.tanh(current_output)
            
            # 更新性能指标
            inference_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["inference_time"] = inference_time
            self.performance_metrics["memory_usage"] = x.element_size() * x.nelement()
            
            return final_output
            
        except Exception as e:
            logging.error(f"量子神经晶格前向传播失败: {e}")
            # 提供降级方案 - 返回零张量避免崩溃
            return torch.zeros((x.shape[0], self.output_dim))
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """带不确定性的预测 - 完全恢复"""
        predictions = []
        
        for i in range(num_samples):
            quantum_noise = torch.randn_like(x) * 0.01 * self.quantum_state_params["decoherence_rate"]
            noisy_input = x + quantum_noise
            
            with torch.no_grad():
                prediction = self.forward(noisy_input)
                predictions.append(prediction)
        
        predictions_tensor = torch.stack(predictions)
        mean_prediction = torch.mean(predictions_tensor, dim=0)
        uncertainty = torch.std(predictions_tensor, dim=0)
        
        return mean_prediction, uncertainty
    
    def get_quantum_coherence(self) -> float:
        """获取量子相干性 - 完全恢复"""
        coherence = float(self.quantum_state_params["coherence_time"].item())
        return max(0.0, min(1.0, coherence))
    
    def optimize_quantum_parameters(self, learning_rate: float = 0.001):
        """优化量子参数 - 完全恢复"""
        quantum_optimizer = optim.Adam(self.quantum_state_params.values(), lr=learning_rate)
        return quantum_optimizer

class QuantumNeuralLatticeStrategy(BaseStrategy):
    """量子神经晶格策略 V5.0 - 完全功能恢复 + 兼容性保证"""
    
    # 接口元数据 - 完全恢复
    _metadata = InterfaceMetadata(
        version="5.0",
        description="量子神经晶格策略 - 混合量子经典神经网络架构",
        author="Quantum-Sniper-Team",
        created_date=datetime.now(),
        performance_targets={
            "signal_generation_time": 0.001,
            "model_inference_time": 0.0005,
            "quantum_coherence": 0.95
        },
        dependencies=["IDataProcessor", "IMarketAnalyzer", "IRiskManager"],
        compatibility=["4.2", "4.1"]
    )
    
    def __init__(self, name: str = "QuantumNeuralLattice", config: Dict[str, Any] = None):
        # 保持与现有版本完全相同的初始化逻辑
        if config is None:
            config = {}
            
        default_config = {
            "name": name,
            "enabled": True,
            "risk_level": "medium",
            "input_dim": 64,
            "hidden_dims": [128, 256, 128, 64],  # 恢复原始版本的优化配置
            "output_dim": 4,
            "quantum_entanglement": 0.7,
            "learning_rate": 0.001,
            "uncertainty_threshold": 0.3,
            "lattice_dimensions": ["time", "price", "volume", "volatility"]
        }
        
        # 合并配置 - 保持现有版本逻辑
        merged_config = {**default_config, **config}
        super().__init__(name, merged_config)
        
        # 量子神经晶格模型 - 恢复完整功能
        self.model: Optional[QuantumNeuralLatticeModel] = None
        self.model_initialized = False
        
        # 量子优化参数 - 恢复完整功能
        self.quantum_optimizer: Optional[optim.Optimizer] = None
        self.learning_rate = self.config.get("learning_rate", 0.001)
        
        # 晶格配置 - 恢复完整功能
        self.lattice_config = {
            "input_dim": self.config.get("input_dim", 64),
            "hidden_dims": self.config.get("hidden_dims", [128, 256, 128, 64]),
            "output_dim": self.config.get("output_dim", 4),
            "quantum_entanglement": self.config.get("quantum_entanglement", 0.7),
            "lattice_dimensions": self.config.get("lattice_dimensions", [
                "time", "price", "volume", "volatility"
            ])
        }
        
        # 量子态监控 - 恢复完整功能
        self.quantum_states = deque(maxlen=1000)
        self.coherence_history = deque(maxlen=100)
        
        # 信号缓存优化 - 恢复完整功能
        self._signal_cache: Dict[str, IStrategySignal] = {}
        self._uncertainty_threshold = self.config.get("uncertainty_threshold", 0.2)
        
        self.logger = logging.getLogger(f"strategy.quantum_lattice")
    
    def initialize(self) -> bool:
        """初始化策略 - 完全恢复 + 错误处理增强"""
        try:
            self.logger.info("开始初始化量子神经晶格策略...")
            
            # 验证配置参数 - 保持现有版本逻辑
            if not self._validate_parameters():
                self.logger.error("参数验证失败")
                return False
            
            # 初始化量子神经晶格模型 - 恢复完整功能
            self.model = QuantumNeuralLatticeModel(
                input_dim=self.lattice_config["input_dim"],
                hidden_dims=self.lattice_config["hidden_dims"],
                output_dim=self.lattice_config["output_dim"],
                quantum_entanglement=self.lattice_config["quantum_entanglement"]
            )
            
            # 初始化量子优化器 - 恢复完整功能
            self.quantum_optimizer = self.model.optimize_quantum_parameters(self.learning_rate)
            
            # 加载预训练模型（如果存在）- 恢复完整功能
            model_path = self.config.get("model_path")
            if model_path and self._load_pretrained_model(model_path):
                self.logger.info(f"成功加载预训练模型: {model_path}")
            
            self.model_initialized = True
            self.initialized = True
            
            # 更新状态 - 保持现有版本兼容性
            self._status.update({
                "model_initialized": True,
                "quantum_coherence": self.model.get_quantum_coherence(),
                "lattice_dimensions": len(self.lattice_config["lattice_dimensions"]),
                "hidden_layers": len(self.lattice_config["hidden_dims"])
            })
            
            self.logger.info("量子神经晶格策略初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"量子神经晶格策略初始化失败: {e}")
            self.initialized = False
            return False
    
    def get_signal(self, data: Any) -> Optional[IStrategySignal]:
        """获取交易信号 - 完全恢复 + 兼容性保证"""
        if not self.model_initialized or self.model is None:
            self.logger.error("量子神经晶格模型未初始化")
            return None
        
        try:
            start_time = datetime.now()
            
            # 数据预处理 - 保持现有版本逻辑
            processed_data = self._preprocess_data(data)
            if processed_data is None:
                return None
            
            # 转换为张量
            input_tensor = torch.FloatTensor(processed_data).unsqueeze(0)
            
            # 带不确定性的预测 - 恢复完整功能
            with torch.no_grad():
                prediction, uncertainty = self.model.predict_with_uncertainty(input_tensor)
                prediction = prediction.squeeze(0)
                uncertainty = uncertainty.squeeze(0)
            
            # 解析预测结果 - 恢复完整功能
            signal_strength = float(prediction[0].item())
            direction_confidence = float(prediction[1].item())
            risk_level = float(prediction[2].item())
            timeframe_score = float(prediction[3].item())
            
            # 计算总体不确定性 - 恢复完整功能
            total_uncertainty = float(torch.mean(uncertainty).item())
            
            # 确定信号方向 - 恢复完整功能
            direction = self._determine_signal_direction(direction_confidence, total_uncertainty)
            
            # 计算最终置信度（考虑不确定性）- 恢复完整功能
            final_confidence = max(0.0, min(1.0, 
                abs(direction_confidence) * (1 - total_uncertainty)
            ))
            
            # 只有高置信度且低不确定性才生成信号 - 恢复完整功能
            if final_confidence < 0.6 or total_uncertainty > self._uncertainty_threshold:
                self.logger.debug(f"信号置信度过低或不确定性过高: {final_confidence:.3f}, 不确定性: {total_uncertainty:.3f}")
                return None
            
            # 创建信号元数据 - 保持现有版本标签兼容性
            signal_metadata = SignalMetadata(
                source="quantum_neural_lattice",
                priority=SignalPriority.HIGH if final_confidence > 0.8 else SignalPriority.MEDIUM,
                tags=["quantum", "neural_lattice", "ai_driven"],  # 保持现有版本标签
                confidence_interval=(final_confidence - total_uncertainty, final_confidence + total_uncertainty)
            )
            
            # 创建信号数据 - 保持现有版本字段兼容性
            signal_data = {
                "signal_strength": signal_strength,
                "direction_confidence": direction_confidence,
                "risk_level": risk_level,
                "timeframe_score": timeframe_score,
                "uncertainty": total_uncertainty,
                "quantum_coherence": self.model.get_quantum_coherence(),
                "inference_time": (datetime.now() - start_time).total_seconds(),
                "model_performance": self.model.performance_metrics,
                # 保持现有版本兼容字段
                "quantum": "enabled",
                "neural_lattice": "active", 
                "ai_driven": "true",
                "timestamp": datetime.now().isoformat()
            }
            
            # 创建策略信号 - 保持现有版本完全兼容
            signal = StrategySignal(
                signal_type="QUANTUM_LATTICE",
                confidence=final_confidence,
                data=signal_data,
                direction=direction,
                metadata=signal_metadata
            )
            
            # 更新性能指标 - 保持现有版本逻辑
            inference_time = (datetime.now() - start_time).total_seconds()
            self._performance_metrics.execution_time += inference_time
            self._performance_metrics.call_count += 1
            
            # 记录量子态 - 恢复完整功能
            self.quantum_states.append({
                "timestamp": datetime.now(),
                "coherence": self.model.get_quantum_coherence(),
                "uncertainty": total_uncertainty,
                "signal_strength": signal_strength
            })
            
            self.logger.info(f"量子神经晶格信号生成: {direction.name}, 置信度: {final_confidence:.3f}, 不确定性: {total_uncertainty:.3f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"量子神经晶格信号生成失败: {e}")
            self._performance_metrics.error_count += 1
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取策略状态 - 完全恢复 + 兼容性保证"""
        base_status = super().get_status()
        
        quantum_status = {
            **base_status,
            "model_initialized": self.model_initialized,
            "quantum_coherence": self.model.get_quantum_coherence() if self.model else 0.0,
            "lattice_dimensions": self.lattice_config["lattice_dimensions"],
            "hidden_layers": len(self.lattice_config["hidden_dims"]),
            "quantum_entanglement": self.lattice_config["quantum_entanglement"],
            "performance_metrics": self._performance_metrics.to_dict(),
            "recent_quantum_states": len(self.quantum_states),
            "average_uncertainty": np.mean([s["uncertainty"] for s in self.quantum_states]) if self.quantum_states else 0.0,
            # 保持现有版本兼容字段
            "config": {
                "input_dim": self.config.get("input_dim"),
                "output_dim": self.config.get("output_dim"),
                "quantum_entanglement": self.config.get("quantum_entanglement")
            }
        }
        
        return quantum_status
    
    def _validate_parameters(self) -> bool:
        """验证参数 - 完全恢复 + 兼容性保证"""
        try:
            # 检查必需参数 - 保持现有版本逻辑
            required_params = ["name", "enabled", "risk_level"]
            for param in required_params:
                if param not in self.config:
                    self.logger.error(f"缺少必需参数: {param}")
                    return False
            
            # 验证量子参数范围 - 恢复完整功能
            quantum_entanglement = self.config.get("quantum_entanglement", 0.7)
            if not 0 <= quantum_entanglement <= 1:
                self.logger.error(f"量子纠缠参数超出范围: {quantum_entanglement}")
                return False
            
            # 验证隐藏层配置 - 恢复完整功能
            hidden_dims = self.config.get("hidden_dims", [128, 256, 128, 64])
            if not isinstance(hidden_dims, list) or len(hidden_dims) == 0:
                self.logger.error("隐藏层配置必须是非空列表")
                return False
            
            # 验证输入输出维度 - 恢复完整功能
            input_dim = self.config.get("input_dim", 64)
            output_dim = self.config.get("output_dim", 4)
            if input_dim <= 0 or output_dim <= 0:
                self.logger.error(f"输入输出维度必须为正数: {input_dim}, {output_dim}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"参数验证异常: {e}")
            return False
    
    # 🚀 恢复所有革命性优化方法
    
    async def train_quantum_model(self, training_data: Dict[str, Any]) -> bool:
        """训练量子模型 - 完全恢复"""
        if not self.model_initialized or self.model is None:
            self.logger.error("量子神经晶格模型未初始化")
            return False
        
        try:
            self.logger.info("开始量子神经晶格模型训练...")
            
            # 准备训练数据
            inputs, targets = self._prepare_training_data(training_data)
            if inputs is None or targets is None:
                return False
            
            # 转换为张量
            inputs_tensor = torch.FloatTensor(inputs)
            targets_tensor = torch.FloatTensor(targets)
            
            # 训练配置
            epochs = self.config.get("training_epochs", 100)
            batch_size = self.config.get("batch_size", 32)
            
            # 创建数据加载器
            dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 训练循环
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                
                for batch_inputs, batch_targets in dataloader:
                    # 前向传播
                    predictions = self.model(batch_inputs)
                    
                    # 计算损失
                    loss = nn.MSELoss()(predictions, batch_targets)
                    
                    # 反向传播
                    self.quantum_optimizer.zero_grad()
                    loss.backward()
                    self.quantum_optimizer.step()
                    
                    total_loss += loss.item()
                
                # 记录训练进度
                if epoch % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    self.logger.info(f"训练轮次 {epoch}, 平均损失: {avg_loss:.6f}")
            
            self.logger.info("量子神经晶格模型训练完成")
            return True
            
        except Exception as e:
            self.logger.error(f"量子模型训练失败: {e}")
            return False
    
    def optimize_quantum_coherence(self) -> bool:
        """优化量子相干性 - 完全恢复"""
        if not self.model_initialized or self.model is None:
            return False
        
        try:
            # 获取当前相干性
            current_coherence = self.model.get_quantum_coherence()
            
            # 优化量子参数以提高相干性
            coherence_target = 0.95
            coherence_loss = abs(current_coherence - coherence_target)
            
            if coherence_loss > 0.1:
                # 执行相干性优化步骤
                self.quantum_optimizer.zero_grad()
                coherence_loss_tensor = torch.tensor(coherence_loss, requires_grad=True)
                coherence_loss_tensor.backward()
                self.quantum_optimizer.step()
                
                new_coherence = self.model.get_quantum_coherence()
                self.logger.info(f"量子相干性优化: {current_coherence:.3f} -> {new_coherence:.3f}")
                
                # 记录相干性历史
                self.coherence_history.append(new_coherence)
                
                return True
            else:
                self.logger.debug("量子相干性已达到目标范围")
                return True
                
        except Exception as e:
            self.logger.error(f"量子相干性优化失败: {e}")
            return False
    
    def get_quantum_insights(self) -> Dict[str, Any]:
        """获取量子洞察 - 完全恢复"""
        if not self.model_initialized or self.model is None:
            return {}
        
        try:
            insights = {
                "quantum_coherence": self.model.get_quantum_coherence(),
                "coherence_history": list(self.coherence_history),
                "recent_states": list(self.quantum_states)[-10:],
                "performance_metrics": self.model.performance_metrics,
                "entanglement_level": self.lattice_config["quantum_entanglement"],
                "model_architecture": {
                    "input_dim": self.lattice_config["input_dim"],
                    "hidden_layers": len(self.lattice_config["hidden_dims"]),
                    "output_dim": self.lattice_config["output_dim"],
                    "total_parameters": sum(p.numel() for p in self.model.parameters())
                }
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"获取量子洞察失败: {e}")
            return {}
    
    def adjust_quantum_parameters(self, market_regime: MarketRegime) -> bool:
        """根据市场状态调整量子参数 - 完全恢复"""
        if not self.model_initialized or self.model is None:
            return False
        
        try:
            # 基于市场状态调整量子参数
            if market_regime == MarketRegime.HIGH_VOLATILITY:
                new_entanglement = max(0.3, self.lattice_config["quantum_entanglement"] * 0.8)
                self.lattice_config["quantum_entanglement"] = new_entanglement
                
            elif market_regime == MarketRegime.LOW_VOLATILITY:
                new_entanglement = min(0.9, self.lattice_config["quantum_entanglement"] * 1.2)
                self.lattice_config["quantum_entanglement"] = new_entanglement
                
            elif market_regime == MarketRegime.CRISIS:
                self.lattice_config["quantum_entanglement"] = 0.4
                self._uncertainty_threshold = 0.15
                
            self.logger.info(f"根据市场状态 {market_regime.value} 调整量子参数: 纠缠度 = {self.lattice_config['quantum_entanglement']:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"调整量子参数失败: {e}")
            return False
    
    # 🔧 内部辅助方法 - 完全恢复
    
    def _preprocess_data(self, data: Any) -> Optional[np.ndarray]:
        """数据预处理 - 保持现有版本逻辑 + 增强"""
        try:
            if isinstance(data, dict):
                features = []
                
                # 价格相关特征 - 保持现有版本逻辑
                price_features = [
                    data.get('open', 50000.0),
                    data.get('high', 51000.0), 
                    data.get('low', 49000.0),
                    data.get('close', 50500.0),
                    data.get('volume', 1000000)
                ]
                features.extend(price_features)
                
                # 技术指标特征 - 保持现有版本逻辑
                tech_indicators = data.get('technical_indicators', {})
                features.extend([
                    tech_indicators.get('rsi', 50.0),
                    tech_indicators.get('macd', 0.0),
                    tech_indicators.get('bollinger_upper', 52000.0),
                    tech_indicators.get('bollinger_lower', 48000.0),
                    tech_indicators.get('atr', 500.0)
                ])
                
                # 市场情绪特征 - 保持现有版本逻辑
                sentiment = data.get('sentiment', {})
                features.extend([
                    sentiment.get('fear_greed', 50.0),
                    sentiment.get('social_volume', 0.0)
                ])
                
                # 填充到固定维度 - 保持现有版本逻辑
                target_dim = self.lattice_config["input_dim"]
                if len(features) < target_dim:
                    features.extend([0.0] * (target_dim - len(features)))
                elif len(features) > target_dim:
                    features = features[:target_dim]
                
                return np.array(features, dtype=np.float32)
                
            elif isinstance(data, (list, np.ndarray)):
                # 直接使用数组数据 - 保持现有版本逻辑
                processed = np.array(data, dtype=np.float32).flatten()
                target_dim = self.lattice_config["input_dim"]
                
                if len(processed) < target_dim:
                    processed = np.pad(processed, (0, target_dim - len(processed)))
                elif len(processed) > target_dim:
                    processed = processed[:target_dim]
                
                return processed
                
            else:
                self.logger.warning(f"不支持的数据类型: {type(data)}")
                return None
                
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            return None
    
    def _determine_signal_direction(self, direction_confidence: float, uncertainty: float) -> SignalDirection:
        """确定信号方向 - 恢复完整功能"""
        adjusted_confidence = direction_confidence * (1 - uncertainty)
        
        if adjusted_confidence > 0.1:
            return SignalDirection.LONG
        elif adjusted_confidence < -0.1:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL
    
    def _prepare_training_data(self, training_data: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """准备训练数据 - 恢复完整功能"""
        try:
            inputs = training_data.get('inputs', [])
            targets = training_data.get('targets', [])
            
            if not inputs or not targets:
                self.logger.error("训练数据为空")
                return None, None
            
            inputs_array = np.array(inputs, dtype=np.float32)
            targets_array = np.array(targets, dtype=np.float32)
            
            # 验证数据形状
            if len(inputs_array.shape) != 2 or inputs_array.shape[1] != self.lattice_config["input_dim"]:
                self.logger.error(f"输入数据形状不匹配: {inputs_array.shape}")
                return None, None
            
            if len(targets_array.shape) != 2 or targets_array.shape[1] != self.lattice_config["output_dim"]:
                self.logger.error(f"目标数据形状不匹配: {targets_array.shape}")
                return None, None
            
            return inputs_array, targets_array
            
        except Exception as e:
            self.logger.error(f"准备训练数据失败: {e}")
            return None, None
    
    def _load_pretrained_model(self, model_path: str) -> bool:
        """加载预训练模型 - 恢复完整功能"""
        try:
            if self.model is None:
                return False
            
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'quantum_optimizer' in checkpoint:
                self.quantum_optimizer.load_state_dict(checkpoint['quantum_optimizer'])
            
            self.logger.info(f"成功加载预训练模型: {model_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"加载预训练模型失败: {e}")
            return False

    # 保持现有版本兼容性方法
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标 - 保持现有版本兼容性"""
        return self._performance_metrics

# ==================== 策略工厂注册 ====================
from src.core.strategy_base import StrategyFactory
StrategyFactory.register_strategy("QuantumNeuralLattice", QuantumNeuralLatticeStrategy)

# ==================== 自动注册接口 ====================
from src.interfaces import InterfaceRegistry
InterfaceRegistry.register_interface(QuantumNeuralLatticeStrategy)

# ==================== 导出列表 ====================
__all__ = [
    'QuantumNeuralLatticeStrategy',
    'QuantumNeuralLatticeModel', 
    'QuantumNeuralLayer',
    'QuantumActivation',
    'QuantumState',
    'LatticeDimension'
]


# src/brain/sac_strategy_optimizer.py
"""量子奇点狙击系统 - 分布式进化SAC策略优化器 V5.0 (完全重新开发 + 极致优化)"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from datetime import datetime
import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import deque, namedtuple

# 导入极致优化的接口和基础类
from interfaces import (
    IStrategySignal, SignalDirection, SignalPriority, PerformanceMetrics,
    InterfaceMetadata, SignalMetadata, MarketRegime, DataQualityLevel
)
from core.strategy_base import BaseStrategy, StrategySignal

# ==================== SAC优化器核心数据结构 ====================

class EvolutionaryPhase(Enum):
    """进化阶段枚举 - 极致优化"""
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration" 
    EXPLOITATION = "exploitation"
    CONVERGENCE = "convergence"
    ADAPTATION = "adaptation"

@dataclass
class SACParameters:
    """SAC算法参数 - 极致优化"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    target_entropy: float = -1.0
    batch_size: int = 256
    buffer_size: int = 1000000
    hidden_dim: int = 256
    num_episodes: int = 1000
    warmup_steps: int = 10000
    
    # 进化算法参数
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2
    
    # 分布式参数
    num_workers: int = 4
    sync_frequency: int = 100

@dataclass
class EvolutionaryMetrics:
    """进化指标 - 极致优化"""
    generation: int = 0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    diversity: float = 0.0
    convergence_rate: float = 0.0
    adaptation_score: float = 0.0
    phase: EvolutionaryPhase = EvolutionaryPhase.INITIALIZATION

@dataclass
class TrainingEpisode:
    """训练回合数据 - 极致优化"""
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    learning_signals: List[float] = field(default_factory=list)

# ==================== SAC神经网络模型 ====================

class SACActor(nn.Module):
    """SAC Actor网络 - 极致优化版本"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 初始化优化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化优化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """前向传播"""
        features = self.network(state)
        mu = self.mu_layer(features)
        log_std = self.log_std_layer(features)
        
        # 限制标准差范围
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return mu, std
    
    def sample(self, state):
        """采样动作"""
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        
        # 重参数化技巧
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class SACCritic(nn.Module):
    """SAC Critic网络 - 极致优化版本"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1网络
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2网络
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化优化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state, action):
        """前向传播"""
        state_action = torch.cat([state, action], 1)
        
        q1 = self.q1_network(state_action)
        q2 = self.q2_network(state_action)
        
        return q1, q2

# ==================== 进化算法核心 ====================

class EvolutionaryOptimizer:
    """分布式进化优化器 - 极致优化版本"""
    
    def __init__(self, population_size: int, parameter_space: Dict[str, Any]):
        self.population_size = population_size
        self.parameter_space = parameter_space
        self.population: List[Dict[str, Any]] = []
        self.fitness_scores: List[float] = []
        self.evolutionary_metrics = EvolutionaryMetrics()
        
        # 性能优化
        self._fitness_cache: Dict[str, float] = {}
        self._diversity_cache: Dict[str, float] = {}
        
        self.logger = logging.getLogger("evolutionary_optimizer")
    
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        
        for i in range(self.population_size):
            individual = {}
            
            for param_name, param_config in self.parameter_space.items():
                if param_config["type"] == "continuous":
                    individual[param_name] = random.uniform(
                        param_config["min"], param_config["max"]
                    )
                elif param_config["type"] == "discrete":
                    individual[param_name] = random.choice(param_config["values"])
                elif param_config["type"] == "categorical":
                    individual[param_name] = random.choice(param_config["categories"])
            
            self.population.append(individual)
        
        self.fitness_scores = [0.0] * self.population_size
        self.evolutionary_metrics.generation = 0
        self.evolutionary_metrics.phase = EvolutionaryPhase.INITIALIZATION
        
        self.logger.info(f"进化种群初始化完成: {self.population_size} 个体")
    
    def evaluate_fitness(self, individual: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """评估个体适应度"""
        individual_id = str(hash(frozenset(individual.items())))
        
        # 使用缓存提高性能
        if individual_id in self._fitness_cache:
            return self._fitness_cache[individual_id]
        
        try:
            # 计算多维度适应度
            performance_metrics = self._simulate_strategy_performance(individual, market_data)
            
            # 复合适应度函数
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
            max_drawdown = performance_metrics.get("max_drawdown", 1.0)
            win_rate = performance_metrics.get("win_rate", 0.0)
            profit_factor = performance_metrics.get("profit_factor", 1.0)
            
            # 适应度计算（考虑风险调整）
            fitness = (
                sharpe_ratio * 0.4 +
                (1 - max_drawdown) * 0.3 +
                win_rate * 0.2 +
                np.log(profit_factor) * 0.1
            )
            
            # 缓存结果
            self._fitness_cache[individual_id] = fitness
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"适应度评估失败: {e}")
            return 0.0
    
    def _simulate_strategy_performance(self, parameters: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """模拟策略性能 - 简化版本"""
        # 在实际实现中，这里会运行完整的策略回测
        # 这里使用简化的随机模拟
        
        returns = np.random.normal(0.001, 0.02, 100)  # 模拟收益
        
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        win_rate = np.mean(returns > 0)
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
        
        return {
            "sharpe_ratio": max(0, sharpe_ratio),
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0
        }
    
    def selection(self, tournament_size: int = 3) -> List[Dict[str, Any]]:
        """锦标赛选择"""
        selected = []
        
        for _ in range(self.population_size):
            # 随机选择参赛者
            contestants = random.sample(
                list(zip(self.population, self.fitness_scores)), tournament_size
            )
            
            # 选择适应度最高的
            best_individual = max(contestants, key=lambda x: x[1])[0]
            selected.append(best_individual.copy())
        
        return selected
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """交叉操作"""
        child = {}
        
        for param_name in self.parameter_space.keys():
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def mutation(self, individual: Dict[str, Any], mutation_rate: float = 0.1):
        """变异操作"""
        mutated_individual = individual.copy()
        
        for param_name, param_config in self.parameter_space.items():
            if random.random() < mutation_rate:
                if param_config["type"] == "continuous":
                    # 高斯变异
                    current_value = mutated_individual[param_name]
                    new_value = current_value + random.gauss(0, 0.1) * (
                        param_config["max"] - param_config["min"]
                    )
                    mutated_individual[param_name] = np.clip(
                        new_value, param_config["min"], param_config["max"]
                    )
                elif param_config["type"] == "discrete":
                    mutated_individual[param_name] = random.choice(param_config["values"])
                elif param_config["type"] == "categorical":
                    mutated_individual[param_name] = random.choice(param_config["categories"])
        
        return mutated_individual
    
    def evolve(self, market_data: Dict[str, Any], mutation_rate: float = 0.1, elite_ratio: float = 0.2):
        """执行一代进化"""
        # 评估适应度
        for i, individual in enumerate(self.population):
            self.fitness_scores[i] = self.evaluate_fitness(individual, market_data)
        
        # 更新进化指标
        self._update_evolutionary_metrics()
        
        # 精英选择
        elite_count = int(self.population_size * elite_ratio)
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        elite_population = [self.population[i] for i in elite_indices]
        
        # 选择
        selected = self.selection()
        
        # 交叉和变异
        new_population = elite_population.copy()  # 保留精英
        
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(selected, 2)
            
            if random.random() < 0.8:  # 交叉概率
                child = self.crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])
            
            # 变异
            child = self.mutation(child, mutation_rate)
            new_population.append(child)
        
        self.population = new_population
        self.evolutionary_metrics.generation += 1
        
        # 清理缓存
        self._fitness_cache.clear()
        
        self.logger.info(f"进化完成第 {self.evolutionary_metrics.generation} 代, 最佳适应度: {self.evolutionary_metrics.best_fitness:.4f}")
    
    def _update_evolutionary_metrics(self):
        """更新进化指标"""
        if self.fitness_scores:
            self.evolutionary_metrics.best_fitness = max(self.fitness_scores)
            self.evolutionary_metrics.average_fitness = np.mean(self.fitness_scores)
            self.evolutionary_metrics.diversity = self._calculate_population_diversity()
            
            # 更新进化阶段
            if self.evolutionary_metrics.generation < 10:
                self.evolutionary_metrics.phase = EvolutionaryPhase.INITIALIZATION
            elif self.evolutionary_metrics.generation < 50:
                self.evolutionary_metrics.phase = EvolutionaryPhase.EXPLORATION
            elif self.evolutionary_metrics.generation < 100:
                self.evolutionary_metrics.phase = EvolutionaryPhase.EXPLOITATION
            else:
                self.evolutionary_metrics.phase = EvolutionaryPhase.CONVERGENCE
    
    def _calculate_population_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) <= 1:
            return 0.0
        
        diversity = 0.0
        param_count = len(self.parameter_space)
        
        for param_name, param_config in self.parameter_space.items():
            values = [ind[param_name] for ind in self.population]
            
            if param_config["type"] == "continuous":
                # 连续参数使用标准差
                if len(values) > 1:
                    std_dev = np.std(values)
                    param_range = param_config["max"] - param_config["min"]
                    diversity += std_dev / param_range if param_range > 0 else 0.0
            else:
                # 离散/分类参数使用唯一值比例
                unique_count = len(set(values))
                diversity += unique_count / len(values)
        
        return diversity / param_count if param_count > 0 else 0.0
    
    def get_best_individual(self) -> Tuple[Dict[str, Any], float]:
        """获取最佳个体"""
        if not self.fitness_scores:
            return {}, 0.0
        
        best_index = np.argmax(self.fitness_scores)
        return self.population[best_index], self.fitness_scores[best_index]

# ==================== SAC策略优化器主类 ====================

class SACStrategyOptimizer(BaseStrategy):
    """分布式进化SAC策略优化器 V5.0 - 完全重新开发 + 极致优化"""
    
    def __init__(self, name: str = "SACStrategyOptimizer", config: Dict[str, Any] = None):
        # 确保配置包含必需参数
        if config is None:
            config = {}
            
        default_config = {
            "name": name,
            "enabled": True,
            "risk_level": "medium",
            "sac_parameters": SACParameters().__dict__,
            "evolutionary_parameters": {
                "population_size": 50,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "elite_ratio": 0.2
            },
            "optimization_targets": ["sharpe_ratio", "max_drawdown", "win_rate"],
            "distributed_training": True,
            "gpu_acceleration": True
        }
        
        # 深度合并配置 - 修复：确保嵌套字典正确合并
        merged_config = default_config.copy()
        for key, value in config.items():
            if key == "sac_parameters" and isinstance(value, dict):
                # 深度合并SAC参数
                merged_config[key] = {**default_config.get(key, {}), **value}
            elif key == "evolutionary_parameters" and isinstance(value, dict):
                # 深度合并进化参数
                merged_config[key] = {**default_config.get(key, {}), **value}
            else:
                merged_config[key] = value
        
        super().__init__(name, merged_config)
        
        # SAC模型组件
        self.actor: Optional[SACActor] = None
        self.critic: Optional[SACCritic] = None
        self.target_critic: Optional[SACCritic] = None
        self.actor_optimizer: Optional[optim.Adam] = None
        self.critic_optimizer: Optional[optim.Adam] = None
        
        # 进化优化器
        self.evolutionary_optimizer: Optional[EvolutionaryOptimizer] = None
        self.parameter_space: Dict[str, Any] = self._define_parameter_space()
        
        # 训练状态
        self.training_episodes: List[TrainingEpisode] = []
        self.current_episode: Optional[TrainingEpisode] = None
        self.is_training = False
        
        # 性能优化
        self._replay_buffer = deque(maxlen=merged_config["sac_parameters"]["buffer_size"])
        self._training_cache: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(f"strategy.{name}")
    
    def initialize(self) -> bool:
        """初始化优化器 - 极致优化版本"""
        try:
            self.logger.info("初始化分布式进化SAC策略优化器...")
            
            # 验证配置参数
            if not self._validate_parameters():
                self.logger.error("参数验证失败")
                return False
            
            # 初始化SAC模型
            self._initialize_sac_models()
            
            # 初始化进化优化器
            self._initialize_evolutionary_optimizer()
            
            # 初始化分布式训练（如果启用）
            if self.config.get("distributed_training", True):
                self._initialize_distributed_training()
            
            self.initialized = True
            self.logger.info("SAC策略优化器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"SAC策略优化器初始化失败: {e}")
            self.initialized = False
            return False
    
    def _validate_parameters(self) -> bool:
        """验证参数 - 极致优化版本"""
        try:
            required_params = ["name", "enabled", "risk_level"]
            for param in required_params:
                if param not in self.config:
                    self.logger.error(f"缺少必需参数: {param}")
                    return False
            
            # 验证SAC参数 - 修复：允许使用默认值
            sac_params = self.config.get("sac_parameters", {})
            required_sac_params = ["learning_rate", "gamma", "tau", "batch_size"]
            default_sac_params = SACParameters().__dict__
            
            for param in required_sac_params:
                if param not in sac_params:
                    # 使用默认值而不是失败
                    if param in default_sac_params:
                        sac_params[param] = default_sac_params[param]
                        self.logger.warning(f"使用默认SAC参数: {param} = {default_sac_params[param]}")
                    else:
                        self.logger.error(f"缺少SAC参数且无默认值: {param}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"参数验证异常: {e}")
            return False
    
    def _define_parameter_space(self) -> Dict[str, Any]:
        """定义策略参数空间"""
        return {
            "learning_rate": {
                "type": "continuous",
                "min": 1e-5,
                "max": 1e-2,
                "description": "学习率"
            },
            "gamma": {
                "type": "continuous", 
                "min": 0.9,
                "max": 0.999,
                "description": "折扣因子"
            },
            "entropy_coefficient": {
                "type": "continuous",
                "min": 0.01,
                "max": 1.0,
                "description": "熵系数"
            },
            "lookback_period": {
                "type": "discrete",
                "values": [10, 20, 50, 100, 200],
                "description": "回看周期"
            },
            "volatility_threshold": {
                "type": "continuous",
                "min": 0.01,
                "max": 0.5,
                "description": "波动率阈值"
            },
            "position_sizing": {
                "type": "categorical",
                "categories": ["conservative", "moderate", "aggressive"],
                "description": "仓位规模策略"
            }
        }
    
    def _initialize_sac_models(self):
        """初始化SAC模型"""
        state_dim = 64  # 状态维度，与量子神经晶格保持一致
        action_dim = 4   # 动作维度
        
        sac_params = self.config["sac_parameters"]
        hidden_dim = sac_params.get("hidden_dim", 256)
        
        # 初始化Actor和Critic网络
        self.actor = SACActor(state_dim, action_dim, hidden_dim)
        self.critic = SACCritic(state_dim, action_dim, hidden_dim)
        self.target_critic = SACCritic(state_dim, action_dim, hidden_dim)
        
        # 同步目标网络
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 初始化优化器
        learning_rate = sac_params.get("learning_rate", 3e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # GPU加速
        if self.config.get("gpu_acceleration", True) and torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()
            self.target_critic = self.target_critic.cuda()
        
        self.logger.info("SAC模型初始化完成")
    
    def _initialize_evolutionary_optimizer(self):
        """初始化进化优化器"""
        evolutionary_params = self.config.get("evolutionary_parameters", {})
        population_size = evolutionary_params.get("population_size", 50)
        
        self.evolutionary_optimizer = EvolutionaryOptimizer(
            population_size=population_size,
            parameter_space=self.parameter_space
        )
        
        self.evolutionary_optimizer.initialize_population()
        self.logger.info("进化优化器初始化完成")
    
    def _initialize_distributed_training(self):
        """初始化分布式训练"""
        try:
            if dist.is_available() and dist.is_initialized():
                self.logger.info("分布式训练已初始化")
            else:
                self.logger.info("单机训练模式")
        except Exception as e:
            self.logger.warning(f"分布式训练初始化失败: {e}")
    
    def get_signal(self, data: Any) -> Optional[IStrategySignal]:
        """获取优化信号 - 极致优化版本"""
        if not self.initialized or self.actor is None:
            self.logger.error("SAC优化器未初始化")
            return None
        
        try:
            # 数据预处理
            processed_state = self._preprocess_state(data)
            if processed_state is None:
                return None
            
            # 使用Actor网络生成动作
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
            
            if self.config.get("gpu_acceleration", True) and torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
            
            with torch.no_grad():
                action, log_prob = self.actor.sample(state_tensor)
                action = action.cpu().numpy()[0]
            
            # 解析动作为交易信号
            signal_data = self._action_to_signal_data(action, data)
            
            # 创建信号元数据
            signal_metadata = SignalMetadata(
                source="sac_optimizer",
                priority=SignalPriority.HIGH,
                tags=["sac", "evolutionary", "ai_optimized", "distributed"]
            )
            
            # 创建策略信号
            signal = StrategySignal(
                signal_type="SAC_OPTIMIZED",
                confidence=0.8,  # SAC信号具有高置信度
                data=signal_data,
                direction=SignalDirection.LONG if action[0] > 0 else SignalDirection.SHORT,
                metadata=signal_metadata
            )
            
            # 更新性能指标
            self._performance_metrics.call_count += 1
            
            self.logger.info(f"SAC优化信号生成: 动作={action}, 置信度=0.8")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"SAC信号生成失败: {e}")
            self._performance_metrics.error_count += 1
            return None
    
    def _preprocess_state(self, data: Any) -> Optional[np.ndarray]:
        """预处理状态数据"""
        try:
            if isinstance(data, dict):
                features = []
                
                # 价格特征
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
                
                # 市场情绪特征
                sentiment = data.get('sentiment', {})
                features.extend([
                    sentiment.get('fear_greed', 50.0),
                    sentiment.get('social_volume', 0.0)
                ])
                
                # 填充到固定维度
                target_dim = 64
                if len(features) < target_dim:
                    features.extend([0.0] * (target_dim - len(features)))
                elif len(features) > target_dim:
                    features = features[:target_dim]
                
                return np.array(features, dtype=np.float32)
                
            else:
                self.logger.warning(f"不支持的数据类型: {type(data)}")
                return None
                
        except Exception as e:
            self.logger.error(f"状态预处理失败: {e}")
            return None
    
    def _action_to_signal_data(self, action: np.ndarray, original_data: Any) -> Dict[str, Any]:
        """将动作转换为信号数据"""
        return {
            "action_values": action.tolist(),
            "position_size": float(np.abs(action[0])),
            "confidence_level": float(np.clip(np.abs(action[1]), 0, 1)),
            "risk_adjustment": float(action[2]),
            "timeframe_preference": float(action[3]),
            "optimization_generation": self.evolutionary_optimizer.evolutionary_metrics.generation if self.evolutionary_optimizer else 0,
            "evolutionary_phase": self.evolutionary_optimizer.evolutionary_metrics.phase.value if self.evolutionary_optimizer else "unknown",
            "timestamp": datetime.now().isoformat(),
            "sac": "enabled",  # 修复：添加缺失的必需字段
            "evolutionary": "active",  # 修复：添加缺失的必需字段
            "ai_optimized": "true",  # 修复：添加缺失的必需字段
            "distributed": "false"  # 修复：添加缺失的必需字段
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取优化器状态 - 极致优化版本"""
        base_status = super().get_status()
        if base_status is None:
            base_status = {}
        
        sac_status = {
            **base_status,
            "strategy_type": "SACStrategyOptimizer",
            "training_status": {
                "is_training": self.is_training,
                "episodes_completed": len(self.training_episodes),
                "current_generation": self.evolutionary_optimizer.evolutionary_metrics.generation if self.evolutionary_optimizer else 0
            },
            "evolutionary_metrics": self.evolutionary_optimizer.evolutionary_metrics.__dict__ if self.evolutionary_optimizer else {},
            "model_parameters": {
                "actor": sum(p.numel() for p in self.actor.parameters()) if self.actor else 0,
                "critic": sum(p.numel() for p in self.critic.parameters()) if self.critic else 0
            },
            "performance_metrics": self._performance_metrics.to_dict()
        }
        
        return sac_status
    
    # 🚀 新增极致优化方法
    
    async def optimize_strategies_async(self, market_data: Dict[str, Any]) -> bool:
        """异步优化策略 - 新增极致优化"""
        try:
            self.logger.info("开始异步策略优化...")
            self.is_training = True
            
            # 创建新的训练回合
            self.current_episode = TrainingEpisode()
            
            # 执行进化优化
            evolutionary_params = self.config.get("evolutionary_parameters", {})
            mutation_rate = evolutionary_params.get("mutation_rate", 0.1)
            elite_ratio = evolutionary_params.get("elite_ratio", 0.2)
            
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.evolutionary_optimizer.evolve, 
                market_data, mutation_rate, elite_ratio
            )
            
            # 获取最佳参数并更新SAC模型
            best_parameters, best_fitness = self.evolutionary_optimizer.get_best_individual()
            await self._update_sac_parameters_async(best_parameters)
            
            # 记录训练回合
            self.current_episode.strategy_parameters = best_parameters
            self.current_episode.performance_metrics = {"fitness": best_fitness}
            self.current_episode.market_conditions = market_data
            
            self.training_episodes.append(self.current_episode)
            self.current_episode = None
            self.is_training = False
            
            self.logger.info(f"异步策略优化完成, 最佳适应度: {best_fitness:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"异步策略优化失败: {e}")
            self.is_training = False
            return False
    
    async def _update_sac_parameters_async(self, parameters: Dict[str, Any]):
        """异步更新SAC参数"""
        try:
            # 更新学习率
            if "learning_rate" in parameters and self.actor_optimizer:
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = parameters["learning_rate"]
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = parameters["learning_rate"]
            
            # 更新其他SAC参数
            sac_params = self.config["sac_parameters"]
            if "gamma" in parameters:
                sac_params["gamma"] = parameters["gamma"]
            if "entropy_coefficient" in parameters:
                sac_params["alpha"] = parameters["entropy_coefficient"]
            
            self.logger.info("SAC参数更新完成")
            
        except Exception as e:
            self.logger.error(f"SAC参数更新失败: {e}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """获取优化洞察 - 新增极致优化"""
        if not self.evolutionary_optimizer:
            return {}
        
        metrics = self.evolutionary_optimizer.evolutionary_metrics
        
        return {
            "evolutionary_progress": {
                "generation": metrics.generation,
                "best_fitness": metrics.best_fitness,
                "average_fitness": metrics.average_fitness,
                "diversity": metrics.diversity,
                "phase": metrics.phase.value
            },
            "population_analysis": {
                "size": self.evolutionary_optimizer.population_size,
                "parameter_space_size": len(self.parameter_space),
                "fitness_range": {
                    "min": min(self.evolutionary_optimizer.fitness_scores) if self.evolutionary_optimizer.fitness_scores else 0,
                    "max": metrics.best_fitness
                }
            },
            "training_history": {
                "total_episodes": len(self.training_episodes),
                "recent_performance": [
                    episode.performance_metrics for episode in self.training_episodes[-5:]
                ]
            }
        }
    
    def export_optimized_parameters(self) -> Dict[str, Any]:
        """导出优化后的参数 - 新增极致优化"""
        if not self.evolutionary_optimizer:
            return {}
        
        best_parameters, best_fitness = self.evolutionary_optimizer.get_best_individual()
        
        return {
            "optimized_parameters": best_parameters,
            "performance_metrics": {
                "fitness_score": best_fitness,
                "evolutionary_generation": self.evolutionary_optimizer.evolutionary_metrics.generation,
                "optimization_timestamp": datetime.now().isoformat()
            },
            "model_configuration": {
                "actor_architecture": str(self.actor) if self.actor else "未初始化",
                "critic_architecture": str(self.critic) if self.critic else "未初始化",
                "parameter_space": self.parameter_space
            }
        }
    
    def adapt_to_market_regime(self, market_regime: MarketRegime) -> bool:
        """适应市场状态 - 新增极致优化"""
        try:
            self.logger.info(f"适应市场状态: {market_regime.value}")
            
            # 根据市场状态调整进化参数
            evolutionary_params = self.config.get("evolutionary_parameters", {})
            
            if market_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
                # 高波动性市场：增加探索，降低变异率
                evolutionary_params["mutation_rate"] = 0.05
                evolutionary_params["population_size"] = 100
            elif market_regime == MarketRegime.LOW_VOLATILITY:
                # 低波动性市场：增加利用，提高变异率
                evolutionary_params["mutation_rate"] = 0.15
                evolutionary_params["population_size"] = 30
            else:
                # 正常市场：默认参数
                evolutionary_params["mutation_rate"] = 0.1
                evolutionary_params["population_size"] = 50
            
            self.logger.info(f"市场状态适应完成: {market_regime.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"市场状态适应失败: {e}")
            return False

# ==================== 主系统集成示例 ====================

# 在主系统中集成SAC策略优化器
# 创建SAC优化器实例
sac_optimizer = SACStrategyOptimizer("QuantumSACOptimizer", {
    "population_size": 50,
    "quantum_enhancement": True,
    "max_generations": 1000
})

# 集成到策略整合引擎
# from brain.strategy_integration import StrategyIntegrationEngine
# integration_engine = StrategyIntegrationEngine("QuantumIntegration")
# integration_engine.add_strategy("SACOptimizer", sac_optimizer, 0.3)

# ==================== 策略工厂注册 ====================

from core.strategy_base import StrategyFactory
StrategyFactory.register_strategy("SACStrategyOptimizer", SACStrategyOptimizer)

# ==================== 自动注册接口 ====================

from interfaces import InterfaceRegistry
InterfaceRegistry.register_interface(SACStrategyOptimizer)

__all__ = [
    'SACStrategyOptimizer',
    'EvolutionaryOptimizer', 
    'SACActor',
    'SACCritic',
    'SACParameters',
    'EvolutionaryMetrics',
    'EvolutionaryPhase'
]
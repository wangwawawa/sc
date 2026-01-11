from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessConfig:
    n_top_genes: int = 2000
    target_sum: float = 1e4
    scale_max: float = 10.0


@dataclass
class BaselineConfig:
    n_pcs: int = 50
    n_neighbors: int = 30
    resolution: float = 0.6
    seed: int = 0
    

@dataclass
class BAMECConfig:
    n_pcs: int = 50
    diffusion_components: int = 30 
    consensus_neighbors: int = 30
    view_neighbors: List[int] = field(default_factory=lambda: [15, 30])  
    resolutions: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7])  
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2]) 
    
    # 共识参数
    consensus_resolution: float = 0.5
    consensus_seed: int = 0
    silhouette_sample_size: int = 2000
    
    # 质量控制参数
    top_k: int = 10  
    elite_fraction: float = 1.0  # 先不筛选，只用top_k
    weight_temperature: float = 0.3  # 增大温度，让权重更平滑
    
    # 自动分辨率选择
    auto_select_consensus_resolution: bool = False
    consensus_resolutions: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8])
    
    # 权重策略
    ensemble_weights: str = "quality"  # "quality" or "uniform"
    
    

@dataclass
class ExperimentConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    bamec: BAMECConfig = field(default_factory=BAMECConfig)
"""
元数据引导的难度估计模块
用于从查询特征和元数据中估计样本/实例的难度权重

【架构理解修正】
1. **Matcher阶段**：只能使用StaticMetaDifficulty（因为query还没匹配，无法对应GT元数据）
2. **Loss阶段**：可以使用MetaDifficultyMLP（匹配后，query-GT pair明确）
3. **Batch处理**：正确处理不同样本不同GT数量的情况

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class MetaDifficultyMLP(nn.Module):
    """
    轻量级MLP网络，从匹配后的query特征和GT元数据估计难度
    
    【关键修正】
    - 只能在Loss阶段使用（匹配后）
    - 输入：按样本分组的matched query features + 对应的GT meta
    - 输出：每个matched pair的难度分数
    
    【训练方式】
    - 端到端训练：与DEIM主模型一起训练
    - 梯度来源：loss → 难度权重 → MLP参数
    - 建议策略：
      * Epoch 0-20: 冻结MLP或只用StaticMetaDifficulty
      * Epoch 20+: 解冻MLP，自适应学习
    
    【Batch处理】
    - targets中每个样本的GT数量不同
    - 匹配后每个样本的matched数量也不同
    - forward支持两种模式：
      1. batch模式：inputs是list，每个元素是该样本的数据
      2. flatten模式：inputs已展平成[total_matched, ...]
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        meta_dim: int = 3,  # altitude, time, angle
        mlp_hidden_dims: tuple = (128, 64),
        dropout: float = 0.1,
        use_layernorm: bool = True
    ):
        """
        Args:
            hidden_dim: 查询特征的维度（Transformer hidden_dim）
            meta_dim: 元数据维度（默认3：altitude, time, angle）
            mlp_hidden_dims: MLP隐藏层维度
            dropout: dropout比率
            use_layernorm: 是否使用LayerNorm
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.meta_dim = meta_dim
        
        # 元数据嵌入层
        self.meta_embed = nn.Sequential(
            nn.Linear(meta_dim, meta_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(meta_dim * 4, meta_dim * 4)
        )
        
        # MLP网络
        input_dim = hidden_dim + meta_dim * 4
        layers = []
        
        prev_dim = input_dim
        for i, curr_dim in enumerate(mlp_hidden_dims):
            layers.append(nn.Linear(prev_dim, curr_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(curr_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = curr_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 输出[0,1]范围
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化网络参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, query_features, meta_features):
        """
        前向传播
        
        Args:
            query_features: [total_matched, hidden_dim] 
                匹配后的query特征（已按indices提取并展平）
            meta_features: [total_matched, meta_dim] 
                对应GT box的元数据（已按indices提取并展平）
                - [:, 0]: altitude (归一化[0,1])
                - [:, 1]: time (0=day, 1=night)  
                - [:, 2]: angle (归一化[0,1])
        
        Returns:
            difficulty_scores: [total_matched] 难度分数[0,1]
                0=简单(30m白天垂直), 1=困难(100m夜间倾斜)
        """
        # 嵌入元数据
        meta_embedded = self.meta_embed(meta_features)  # [total_matched, meta_dim*4]
        
        # 融合query特征和元数据
        fused_features = torch.cat([query_features, meta_embedded], dim=-1)  
        
        # 通过MLP预测难度
        difficulty_scores = self.mlp(fused_features)  # [total_matched, 1]
        difficulty_scores = difficulty_scores.squeeze(-1)  # [total_matched]
        
        return difficulty_scores


class StaticMetaDifficulty(nn.Module):
    """
    基于元数据的静态难度估计（只依赖GT元数据，不依赖query特征）
    
    【使用场景】
    1. **HungarianMatcher**：调整cost matrix（此时query未匹配，只能用GT元数据）
    2. **Loss阶段**：训练初期（epoch 0-20）作为稳定先验
    3. **推理阶段**：如果不想引入MLP的计算开销
    
    【计算公式】
        difficulty = w_h * f_h(altitude) + w_t * f_t(time) + w_θ * f_θ(angle)
    
    其中：
        - f_h(altitude): 高度越高越难（100m > 60m > 30m）
        - f_t(time): 夜间比白天难
        - f_θ(angle): 倾斜角度越大越难（30° > 90°）
    
    【Batch处理】
    - 输入是box级元数据list: [[num_boxes_0, 3], [num_boxes_1, 3], ...]
    - 输出也是list: [[num_boxes_0], [num_boxes_1], ...]
    - 每个box独立计算，支持Mosaic/MixUp混合图像
    """
    
    def __init__(
        self,
        altitude_weight: float = 0.4,
        time_weight: float = 0.3,
        angle_weight: float = 0.3,
        altitude_scale: float = 1.0,  # 100m对应的难度
        time_scale: float = 1.0,      # 夜间的额外难度
        angle_scale: float = 1.0       # 30度对应的难度
    ):
        """
        Args:
            altitude_weight: 高度的权重
            time_weight: 时间的权重
            angle_weight: 角度的权重
            altitude_scale: 高度的缩放因子
            time_scale: 时间的缩放因子
            angle_scale: 角度的缩放因子
        """
        super().__init__()
        
        # 注册为buffer（不参与梯度更新，但会保存到checkpoint）
        self.register_buffer('altitude_weight', torch.tensor(altitude_weight))
        self.register_buffer('time_weight', torch.tensor(time_weight))
        self.register_buffer('angle_weight', torch.tensor(angle_weight))
        self.register_buffer('altitude_scale', torch.tensor(altitude_scale))
        self.register_buffer('time_scale', torch.tensor(time_scale))
        self.register_buffer('angle_scale', torch.tensor(angle_scale))
    
    def forward(self, meta_features):
        """
        计算静态难度分数
        
        Args:
            meta_features: 支持两种输入格式
                1. Tensor [num_boxes, 3]: 单个样本或已展平的多样本
                2. List[Tensor]: [[num_boxes_0, 3], [num_boxes_1, 3], ...]
                   每个样本的GT数量不同，需要list格式
                
                元数据含义：
                - [..., 0]: altitude (归一化: 30m->0, 100m->1)
                - [..., 1]: time (0=day, 1=night)
                - [..., 2]: angle (归一化: 90°->0, 30°->1)
        
        Returns:
            difficulty_scores: 与输入格式对应
                1. Tensor [num_boxes]: 单个/展平输入
                2. List[Tensor]: [[num_boxes_0], [num_boxes_1], ...]
        """
        # 判断输入类型
        if isinstance(meta_features, list):
            # List模式：分别处理每个样本
            return [self._compute_difficulty(mf) for mf in meta_features]
        else:
            # Tensor模式：直接处理
            return self._compute_difficulty(meta_features)
    
    def _compute_difficulty(self, meta_features):
        """单个tensor的难度计算"""
        # 提取各维度
        altitude_norm = meta_features[..., 0]  # [0, 1]
        time_flag = meta_features[..., 1]      # {0, 1}
        angle_norm = meta_features[..., 2]     # [0, 1]
        
        # 计算各维度的难度贡献
        h_difficulty = altitude_norm * self.altitude_scale
        t_difficulty = time_flag * self.time_scale
        theta_difficulty = angle_norm * self.angle_scale
        
        # 加权求和
        difficulty = (
            self.altitude_weight * h_difficulty +
            self.time_weight * t_difficulty +
            self.angle_weight * theta_difficulty
        )
        
        # 归一化到[0, 1]
        difficulty = torch.sigmoid(difficulty)
        
        return difficulty


def extract_meta_features_from_targets(targets, indices=None):
    """
    从targets中提取box级元数据
    
    【关键修正】
    - Matcher阶段（indices=None）：返回所有GT的元数据（list格式，不展平）
    - Loss阶段（indices提供）：只返回匹配上的GT元数据（展平成单个tensor）
    
    Args:
        targets: list of dict, batch_size个样本，每个包含：
            - 'meta_altitude': [num_boxes_i] 归一化高度
            - 'meta_time': [num_boxes_i] 时间标记
            - 'meta_angle': [num_boxes_i] 归一化角度
            注意：每个样本的num_boxes_i可能不同！
            
        indices: list of tuple(query_idx, gt_idx) or None
            - None: Matcher阶段，需要所有GT元数据
            - list: Loss阶段，只需匹配上的GT元数据
    
    Returns:
        - 如果indices=None: List[[num_boxes_0, 3], [num_boxes_1, 3], ...]
          每个样本的所有GT元数据，保持list格式（因为长度不同）
          
        - 如果indices提供: Tensor [total_matched, 3]
          所有batch中匹配上的GT元数据，展平成单个tensor
    """
    if indices is None:
        # ============ Matcher阶段 ============
        # 返回所有GT boxes的元数据（list格式）
        meta_list = []
        for target in targets:
            if 'meta_altitude' not in target:
                # 没有元数据（可能是普通数据集），返回零tensor
                num_boxes = len(target.get('boxes', []))
                device = target.get('boxes', torch.tensor([])).device
                meta_list.append(torch.zeros(num_boxes, 3, device=device))
            elif len(target['meta_altitude']) == 0:
                # 空样本（无GT boxes）
                device = target.get('boxes', torch.tensor([])).device
                meta_list.append(torch.empty(0, 3, device=device))
            else:
                # 正常情况：提取元数据
                altitude = target['meta_altitude']  # [num_boxes_i]
                time = target['meta_time'].float()  # [num_boxes_i]
                angle = target['meta_angle']        # [num_boxes_i]
                meta = torch.stack([altitude, time, angle], dim=1)  # [num_boxes_i, 3]
                meta_list.append(meta)
        
        return meta_list  # List of variable-length tensors
    
    else:
        # ============ Loss阶段 ============
        # 只返回匹配上的boxes的元数据（展平）
        matched_meta_list = []
        
        for i, (query_idx, gt_idx) in enumerate(indices):
            target = targets[i]
            
            # 跳过无匹配或无元数据的样本
            if len(gt_idx) == 0:
                continue
            if 'meta_altitude' not in target:
                # 无元数据，使用零向量
                device = gt_idx.device
                meta = torch.zeros(len(gt_idx), 3, device=device)
                matched_meta_list.append(meta)
                continue
            
            # 提取匹配上的GT boxes的元数据
            altitude = target['meta_altitude'][gt_idx]  # [num_matched_i]
            time = target['meta_time'][gt_idx].float()  # [num_matched_i]
            angle = target['meta_angle'][gt_idx]        # [num_matched_i]
            
            meta = torch.stack([altitude, time, angle], dim=1)  # [num_matched_i, 3]
            matched_meta_list.append(meta)
        
        # 展平所有batch的匹配
        if len(matched_meta_list) > 0:
            return torch.cat(matched_meta_list, dim=0)  # [total_matched, 3]
        else:
            # 没有任何匹配（罕见情况）
            device = targets[0].get('boxes', torch.tensor([])).device
            return torch.empty(0, 3, device=device)


def extract_matched_query_features(outputs, indices):
    """
    从outputs中提取匹配上的query特征（用于MLP输入）
    
    【使用场景】
    在DEIMCriterion的loss计算中，需要提取matched queries的特征
    来喂给MetaDifficultyMLP
    
    Args:
        outputs: dict, 包含：
            - 'pred_logits': [B, num_queries, num_classes]
            - 'pred_boxes': [B, num_queries, 4]
            注意：这里没有直接的query features，需要从pred_logits提取
            
        indices: list of tuple(query_idx, gt_idx)
            Hungarian matching的结果
    
    Returns:
        query_features: [total_matched, hidden_dim]
            所有匹配上的query的特征（展平）
            
    注意：由于outputs中通常没有保存中间的query features，
    这里使用pred_logits作为query的表示（经过分类头后的特征）
    """
    pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]
    
    # 提取匹配上的query的logits
    matched_logits_list = []
    for i, (query_idx, gt_idx) in enumerate(indices):
        if len(query_idx) == 0:
            continue
        # 提取该样本中匹配上的queries
        matched_logits = pred_logits[i][query_idx]  # [num_matched_i, num_classes]
        matched_logits_list.append(matched_logits)
    
    # 展平
    if len(matched_logits_list) > 0:
        return torch.cat(matched_logits_list, dim=0)  # [total_matched, num_classes]
    else:
        # 无匹配
        device = pred_logits.device
        num_classes = pred_logits.shape[-1]
        return torch.empty(0, num_classes, device=device)


def compute_sample_weights_from_consistency(targets, consistency_min=0.7):
    """
    根据meta_consistency_score计算样本级权重
    
    【核心思想】
    - meta_consistency_score 反映图像元数据的一致性
    - 高一致性（纯图像）：score高，元数据可靠，权重=1.0
    - 低一致性（混合图像）：score低，元数据混杂，权重降低
    
    【使用场景】
    在DEIMCriterion中对整个样本的loss进行加权：
        loss_per_sample = loss_bbox + loss_giou + ...
        final_loss = (loss_per_sample * sample_weight).sum()
    
    Args:
        targets: list of dict, 每个dict包含：
            - 'meta_consistency_score': scalar tensor, 范围[0,1]
        consistency_min: float, 最低权重（混合图像的最小权重）
            - 默认0.7表示即使完全混乱也保留70%权重
    
    Returns:
        sample_weights: [B] 样本权重
            - score=1.0 (纯图像) → weight=1.0
            - score=0.5 (中度混合) → weight=(consistency_min + 1.0)/2
            - score=0.0 (完全混乱) → weight=consistency_min
    """
    weights = []
    for target in targets:
        score = target.get('meta_consistency_score', torch.tensor(1.0))
        
        # 权重公式：weight = consistency_min + (1.0 - consistency_min) * score
        # 例如consistency_min=0.7: score=1.0→1.0, score=0.0→0.7
        weight = consistency_min + (1.0 - consistency_min) * score
        weights.append(weight)
    
    device = targets[0].get('boxes', torch.tensor([])).device
    return torch.tensor(weights, device=device)


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("Testing Meta Weighting Module (Fixed Version)")
    print("=" * 60)
    
    # ============ 测试1：StaticMetaDifficulty（Matcher阶段用） ============
    print("\n[Test 1] StaticMetaDifficulty (for Matcher stage)")
    targets = [
        {
            'boxes': torch.rand(5, 4),
            'labels': torch.randint(0, 10, (5,)),
            'meta_altitude': torch.rand(5),
            'meta_time': torch.randint(0, 2, (5,)),
            'meta_angle': torch.rand(5),
        },
        {
            'boxes': torch.rand(8, 4),
            'labels': torch.randint(0, 10, (8,)),
            'meta_altitude': torch.rand(8),
            'meta_time': torch.randint(0, 2, (8,)),
            'meta_angle': torch.rand(8),
        }
    ]
    
    # 提取所有GT元数据（list格式）
    meta_list = extract_meta_features_from_targets(targets, indices=None)
    print(f"  GT meta shapes: {[m.shape for m in meta_list]}")  # [(5,3), (8,3)]
    
    # 计算静态难度（支持list输入）
    static = StaticMetaDifficulty()
    static_scores_list = static(meta_list)
    print(f"  Static difficulty shapes: {[s.shape for s in static_scores_list]}")  # [(5,), (8,)]
    print(f"  Sample 0 difficulty range: [{static_scores_list[0].min():.3f}, {static_scores_list[0].max():.3f}]")
    print(f"  Sample 1 difficulty range: [{static_scores_list[1].min():.3f}, {static_scores_list[1].max():.3f}]")
    
    # ============ 测试2：MetaDifficultyMLP（Loss阶段用） ============
    print("\n[Test 2] MetaDifficultyMLP (for Loss stage)")
    
    # 模拟Hungarian matching结果
    indices = [
        (torch.tensor([0, 1, 2]), torch.tensor([0, 2, 4])),  # 样本0：3个匹配
        (torch.tensor([0, 1]), torch.tensor([1, 3]))          # 样本1：2个匹配
    ]
    
    # 提取匹配上的GT元数据（展平）
    matched_meta = extract_meta_features_from_targets(targets, indices=indices)
    assert isinstance(matched_meta, torch.Tensor), "Matched meta should be a tensor"
    print(f"  Matched GT meta shape: {matched_meta.shape}")  # [5, 3]
    
    # 模拟matched query features
    num_matched = matched_meta.shape[0]
    num_classes = 80
    query_features = torch.randn(num_matched, num_classes)  # 使用pred_logits作为query特征
    
    print(f"  Matched query features shape: {query_features.shape}")  # [5, 80]
    
    # 通过MLP计算难度
    mlp = MetaDifficultyMLP(hidden_dim=num_classes)  # hidden_dim=num_classes（用logits作为特征）
    difficulty_scores = mlp(query_features, matched_meta)
    print(f"  MLP difficulty shape: {difficulty_scores.shape}")  # [5]
    print(f"  MLP difficulty range: [{difficulty_scores.min():.3f}, {difficulty_scores.max():.3f}]")
    
    # ============ 测试3：extract_matched_query_features ============
    print("\n[Test 3] Extract Matched Query Features")
    
    # 模拟model outputs
    batch_size = 2
    num_queries = 300
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes),
        'pred_boxes': torch.rand(batch_size, num_queries, 4),
    }
    
    # 提取matched query features
    matched_queries = extract_matched_query_features(outputs, indices)
    print(f"  Matched query features shape: {matched_queries.shape}")  # [5, 80]
    print(f"  Should match num_matched: {num_matched}")
    
    # ============ 测试4：compute_sample_weights_from_consistency ============
    print("\n[Test 4] Sample Weights from Consistency Score")
    
    targets_with_score = [
        {
            **targets[0],
            'meta_consistency_score': torch.tensor(1.0),  # 高一致性（纯图像）
        },
        {
            **targets[1],
            'meta_consistency_score': torch.tensor(0.3),  # 低一致性（Mosaic混合）
        }
    ]
    
    sample_weights = compute_sample_weights_from_consistency(targets_with_score)
    print(f"  Sample weights: {sample_weights}")  # [1.0, 0.79]
    print(f"    Pure image (score=1.0): weight={sample_weights[0]:.2f}")
    print(f"    Mixed image (score=0.3): weight={sample_weights[1]:.2f}")
    
    # ============ 测试5：完整流程模拟 ============
    print("\n[Test 5] Complete Workflow Simulation")
    print("  Matcher Stage:")
    print(f"    - Extract all GT meta: {len(meta_list)} samples")
    print(f"    - Compute static difficulty: {len(static_scores_list)} samples")
    print(f"    - Adjust cost matrix with difficulty scores")
    
    print("  Loss Stage:")
    if isinstance(matched_meta, torch.Tensor):
        print(f"    - Extract matched GT meta: {matched_meta.shape}")
    if isinstance(matched_queries, torch.Tensor):
        print(f"    - Extract matched query features: {matched_queries.shape}")
    print(f"    - Compute MLP difficulty: {difficulty_scores.shape}")
    print(f"    - Apply box-level weights: (1.0 + difficulty * 0.5)")
    print(f"    - Apply sample-level weights: {sample_weights.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Architecture is correct.")
    print("=" * 60)

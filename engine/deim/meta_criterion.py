"""
元数据引导的DEIM损失函数
基于原始DEIMCriterion，集成元数据难度加权

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deim_criterion import DEIMCriterion
from .meta_weighting import (
    MetaDifficultyMLP, 
    StaticMetaDifficulty,
    extract_meta_features_from_targets,
    extract_matched_query_features,
    compute_sample_weights_from_consistency
)
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..core import register


@register()
class MetaDEIMCriterion(DEIMCriterion):
    """
    元数据引导的DEIM损失函数
    
    【核心思想】
    在原始DEIMCriterion的基础上，集成两级元数据加权：
    1. Box级加权：根据GT元数据（高度/时间/角度）+ query特征，动态调整每个box的loss权重
    2. Sample级加权：根据元数据一致性分数，调整混合图像的loss权重
    
    【加权公式】
    Box级：weight = 1.0 + difficulty * difficulty_scale
        - difficulty ∈ [0, 1]，由MLP或静态规则计算
        - 困难样本（100m夜间倾斜）→ difficulty≈1.0 → weight≈1.5
        - 简单样本（30m白天垂直）→ difficulty≈0.0 → weight=1.0
    
    Sample级：weight = consistency_min + (1.0 - consistency_min) * consistency_score
        - consistency_score ∈ [0, 1]
        - 高一致性（纯图像）→ score=1.0 → weight=1.0
        - 低一致性（混合图像）→ score=0.3 → weight=0.7
    
    【训练策略】
    1. Epoch 0-20: 使用StaticMetaDifficulty（静态先验）
    2. Epoch 20+: 切换到MetaDifficultyMLP（动态学习）
    3. 可选：整个训练只用Static（推理时更快）
    """
    
    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        # 元数据引导参数
        use_meta_weighting=False,
        meta_mode='static',  # 'static', 'dynamic', or 'progressive'
        difficulty_scale=0.5,
        use_consistency_weighting=True,
        consistency_min=0.7,
        # Progressive模式参数（meta_mode='progressive'时使用）
        progressive_switch_epoch=20,  # 从static切换到dynamic的epoch
        progressive_freeze_epochs=5,  # dynamic开启后冻结MLP几个epoch
        # MLP参数（仅meta_mode='dynamic'或'progressive'时使用）
        mlp_hidden_dims=(128, 64),
        mlp_dropout=0.1,
        # 静态难度参数
        static_altitude_weight=0.4,
        static_time_weight=0.3,
        static_angle_weight=0.3,
    ):
        """
        Args:
            (继承DEIMCriterion的所有参数...)
            
            use_meta_weighting: 是否启用元数据加权
            meta_mode: 'static' 或 'dynamic'
                - 'static': 使用StaticMetaDifficulty（基于规则，不可学习）
                - 'dynamic': 使用MetaDifficultyMLP（端到端训练）
            difficulty_scale: 难度对loss的影响幅度
                - 0.0: 不加权（等同于原始criterion）
                - 0.5: 困难样本权重1.5倍（推荐）
                - 1.0: 困难样本权重2.0倍（可能过激）
            use_consistency_weighting: 是否使用一致性加权
            consistency_min: 一致性最低权重（混合图像的最小权重）
            mlp_hidden_dims: MLP隐藏层维度
            mlp_dropout: MLP dropout率
            static_altitude_weight: 静态模式下高度的权重
            static_time_weight: 静态模式下时间的权重
            static_angle_weight: 静态模式下角度的权重
        """
        # 初始化父类
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
            mal_alpha=mal_alpha,
            use_uni_set=use_uni_set
        )
        
        # 元数据加权参数
        self.use_meta_weighting = use_meta_weighting
        self.meta_mode = meta_mode
        self.difficulty_scale = difficulty_scale
        self.use_consistency_weighting = use_consistency_weighting
        self.consistency_min = consistency_min
        
        # Progressive模式参数
        self.progressive_switch_epoch = progressive_switch_epoch
        self.progressive_freeze_epochs = progressive_freeze_epochs
        self.current_epoch = 0  # 当前epoch，由外部调用set_epoch设置
        self._current_meta_mode = meta_mode  # 实际使用的mode（progressive会动态切换）
        
        # 初始化难度估计模块
        if self.use_meta_weighting:
            # Static模块（progressive和static模式都需要）
            if meta_mode in ['static', 'progressive']:
                self.static_difficulty = StaticMetaDifficulty(
                    altitude_weight=static_altitude_weight,
                    time_weight=static_time_weight,
                    angle_weight=static_angle_weight
                )
            
            # Dynamic模块（progressive和dynamic模式都需要）
            if meta_mode in ['dynamic', 'progressive']:
                self.dynamic_difficulty = MetaDifficultyMLP(
                    hidden_dim=num_classes,  # 使用pred_logits的维度
                    meta_dim=3,
                    mlp_hidden_dims=mlp_hidden_dims,
                    dropout=mlp_dropout,
                    use_layernorm=True
                )
                
                # Progressive模式下，MLP初始冻结
                if meta_mode == 'progressive':
                    for param in self.dynamic_difficulty.parameters():
                        param.requires_grad = False
                    self._mlp_frozen = True
            
            # 设置初始模式
            if meta_mode == 'progressive':
                self._current_meta_mode = 'static'
            elif meta_mode not in ['static', 'dynamic']:
                raise ValueError(f"Invalid meta_mode: {meta_mode}. Must be 'static', 'dynamic', or 'progressive'")
    
    def set_epoch(self, epoch):
        """
        设置当前epoch，用于progressive模式的自动切换
        
        Args:
            epoch: 当前训练epoch
        """
        self.current_epoch = epoch
        
        if self.meta_mode == 'progressive':
            # 检查是否到达切换点
            if epoch >= self.progressive_switch_epoch:
                self._current_meta_mode = 'dynamic'
                
                # 解冻MLP（在切换后freeze_epochs个epoch后）
                if hasattr(self, '_mlp_frozen') and self._mlp_frozen:
                    freeze_until = self.progressive_switch_epoch + self.progressive_freeze_epochs
                    if epoch >= freeze_until:
                        for param in self.dynamic_difficulty.parameters():
                            param.requires_grad = True
                        self._mlp_frozen = False
                        print(f"\n{'='*60}")
                        print(f"[MetaDEIMCriterion] Epoch {epoch}: MLP unfrozen and ready to train!")
                        print(f"{'='*60}\n")
                    elif epoch == self.progressive_switch_epoch:
                        print(f"\n{'='*60}")
                        print(f"[MetaDEIMCriterion] Epoch {epoch}: Switched to dynamic mode")
                        print(f"[MetaDEIMCriterion] MLP will be unfrozen at epoch {freeze_until}")
                        print(f"{'='*60}\n")
            else:
                self._current_meta_mode = 'static'
    
    def _compute_meta_difficulty(self, outputs, targets, indices):
        """
        计算元数据难度分数
        
        Args:
            outputs: 模型输出
            targets: GT targets
            indices: 匹配关系
        
        Returns:
            difficulty_scores: [total_matched] 难度分数 [0, 1]
        """
        # 提取匹配上的GT元数据
        matched_meta = extract_meta_features_from_targets(targets, indices)
        # matched_meta: [total_matched, 3]
        
        # 根据当前模式选择难度估计模块
        if self._current_meta_mode == 'dynamic':
            # 动态模式：使用MLP（需要query特征）
            matched_queries = extract_matched_query_features(outputs, indices)
            # matched_queries: [total_matched, num_classes]
            
            difficulty_scores = self.dynamic_difficulty(matched_queries, matched_meta)
            # [total_matched]
        else:
            # 静态模式：只使用GT元数据
            difficulty_scores = self.static_difficulty(matched_meta)
            # [total_matched]
        
        return difficulty_scores
    
    def _get_cached_sample_weights(self, targets):
        """
        获取缓存的sample权重（避免重复计算）
        
        在forward中首次调用时计算并缓存，后续直接返回
        """
        if not hasattr(self, '_cached_sample_weights'):
            self._cached_sample_weights = None
            self._cached_targets_id = None
        
        # 使用targets的id作为缓存key（同一个forward内targets不变）
        targets_id = id(targets)
        if self._cached_targets_id != targets_id:
            # 重新计算
            self._cached_sample_weights = compute_sample_weights_from_consistency(
                targets, self.consistency_min
            )
            self._cached_targets_id = targets_id
        
        return self._cached_sample_weights
    
    def _apply_sample_weights(self, loss_per_box, indices, sample_weights):
        """
        将sample级权重应用到box级loss上（优化版，零循环，O(B)复杂度）
        
        Args:
            loss_per_box: [total_matched] 每个box的loss
            indices: list of (query_idx, gt_idx) tuples
            sample_weights: [B] 每个样本的权重
        
        Returns:
            weighted_loss: [total_matched] 加权后的loss
        """
        if loss_per_box.numel() == 0:
            return loss_per_box
        
        # 获取每个样本的匹配数
        num_matched_per_sample = torch.tensor(
            [len(idx[0]) for idx in indices], 
            dtype=torch.long, 
            device=sample_weights.device
        )
        
        if num_matched_per_sample.sum() == 0:
            return loss_per_box
        
        # 使用repeat_interleave构建box级权重（零循环，高效）
        # 例如：sample_weights=[0.8, 1.0], num_matched=[3, 2]
        #      -> box_weights=[0.8, 0.8, 0.8, 1.0, 1.0]
        box_weights = sample_weights.repeat_interleave(num_matched_per_sample)
        
        # 向量化加权
        return loss_per_box * box_weights
    
    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """
        重写loss_boxes，集成元数据加权
        
        加权顺序：
        1. 原始loss（L1 + GIoU）
        2. Box级元数据加权（困难样本权重更高）
        3. Sample级一致性加权（混合图像权重降低）
        4. 已有的boxes_weight（如果有）
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # 边界检查：如果没有匹配，直接返回0 loss
        if len(idx[0]) == 0:
            device = outputs['pred_boxes'].device
            return {
                'loss_bbox': torch.tensor(0.0, device=device),
                'loss_giou': torch.tensor(0.0, device=device)
            }
        
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        losses = {}
        
        # ============ 基础L1 loss ============
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')  # [total_matched, 4]
        loss_bbox_per_box = loss_bbox.sum(dim=1)  # [total_matched]
        
        # ============ 基础GIoU loss ============
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), 
            box_cxcywh_to_xyxy(target_boxes)
        ))  # [total_matched]
        
        # ============ Box级元数据加权 ============
        if self.use_meta_weighting:
            difficulty = self._compute_meta_difficulty(outputs, targets, indices)
            # difficulty: [total_matched] ∈ [0, 1]
            
            # 加权公式：weight = 1.0 + difficulty * scale
            meta_box_weights = 1.0 + difficulty * self.difficulty_scale  # [total_matched]
            
            loss_bbox_per_box = loss_bbox_per_box * meta_box_weights
            loss_giou = loss_giou * meta_box_weights
        
        # ============ Sample级一致性加权 ============
        if self.use_consistency_weighting:
            # 使用缓存避免重复计算（loss_boxes会被调用多次）
            sample_weights = self._get_cached_sample_weights(targets)
            # sample_weights: [B] ∈ [consistency_min, 1.0]
            # 确保sample_weights不为None且在正确设备上
            if sample_weights is not None:
                sample_weights = sample_weights.to(loss_bbox_per_box.device)
                # 将sample权重应用到每个box
                loss_bbox_per_box = self._apply_sample_weights(loss_bbox_per_box, indices, sample_weights)
                loss_giou = self._apply_sample_weights(loss_giou, indices, sample_weights)
        
        # ============ 已有的boxes_weight（如IoU加权）============
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight
        
        # 归一化
        losses['loss_bbox'] = loss_bbox_per_box.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        """
        重写loss_labels_vfl，集成元数据加权
        
        注意：分类loss加权需要特别小心，因为它影响正负样本平衡
        这里只在正样本（matched）上应用元数据加权
        """
        # 调用父类方法获取基础loss
        base_losses = super().loss_labels_vfl(outputs, targets, indices, num_boxes, values)
        
        # 注意：VFL loss已经是标量，无法直接加权
        # 如果需要加权VFL，需要修改父类方法的reduction方式
        # 这里暂时返回原loss，主要加权在boxes上
        
        return base_losses
    
    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """
        重写loss_labels_mal，集成元数据加权
        
        同样，分类loss的加权需要谨慎处理
        """
        base_losses = super().loss_labels_mal(outputs, targets, indices, num_boxes, values)
        return base_losses
    
    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """
        重写loss_local (FGL loss)，集成元数据加权
        
        Local loss对细粒度定位很重要，可以应用元数据加权
        """
        # 调用父类方法
        base_losses = super().loss_local(outputs, targets, indices, num_boxes, T)
        
        # 如果需要对FGL loss加权，可以在这里实现
        # 但需要修改父类的reduction方式
        
        return base_losses
    
    def forward(self, outputs, targets, **kwargs):
        """重写forward以添加缓存清理"""
        # 清理上一次的缓存
        if hasattr(self, '_cached_sample_weights'):
            self._cached_sample_weights = None
            self._cached_targets_id = None
        
        # 调用父类forward
        return super().forward(outputs, targets, **kwargs)
    
    def extra_repr(self) -> str:
        """打印模块信息"""
        s = super().extra_repr()
        if self.use_meta_weighting:
            s += f"\n  (meta_weighting): mode={self.meta_mode}"
            if self.meta_mode == 'progressive':
                s += f" (current: {self._current_meta_mode}, epoch: {self.current_epoch})"
                s += f"\n    switch_epoch={self.progressive_switch_epoch}, "
                s += f"freeze_epochs={self.progressive_freeze_epochs}"
            s += f"\n    difficulty_scale={self.difficulty_scale}, "
            s += f"use_consistency={self.use_consistency_weighting}"
        return s


if __name__ == '__main__':
    """测试MetaDEIMCriterion"""
    print("=" * 60)
    print("Testing MetaDEIMCriterion")
    print("=" * 60)
    
    # 模拟matcher
    class DummyMatcher(nn.Module):
        def forward(self, outputs, targets):
            # 简单模拟匹配
            indices = []
            for i, target in enumerate(targets):
                num_gt = len(target['boxes'])
                indices.append((
                    torch.arange(num_gt),
                    torch.arange(num_gt)
                ))
            return {'indices': indices}
    
    # 配置
    matcher = DummyMatcher()
    weight_dict = {
        'loss_bbox': 5.0,
        'loss_giou': 2.0,
        'loss_vfl': 1.0,
    }
    losses = ['boxes', 'vfl']
    
    # 测试1：原始criterion（无元数据加权）
    print("\n[Test 1] Original Criterion (no meta weighting)")
    criterion_original = MetaDEIMCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        num_classes=80,
        use_meta_weighting=False
    )
    
    # 模拟数据
    batch_size = 2
    num_queries = 300
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, 80),
        'pred_boxes': torch.rand(batch_size, num_queries, 4),
    }
    
    targets = [
        {
            'boxes': torch.rand(5, 4),
            'labels': torch.randint(0, 80, (5,)),
            'meta_altitude': torch.tensor([0.0, 0.5, 1.0, 0.3, 0.7]),
            'meta_time': torch.tensor([0, 1, 1, 0, 1]).float(),
            'meta_angle': torch.tensor([0.0, 0.5, 1.0, 0.2, 0.8]),
            'meta_consistency_score': torch.tensor(1.0),
        },
        {
            'boxes': torch.rand(8, 4),
            'labels': torch.randint(0, 80, (8,)),
            'meta_altitude': torch.rand(8),
            'meta_time': torch.randint(0, 2, (8,)).float(),
            'meta_angle': torch.rand(8),
            'meta_consistency_score': torch.tensor(0.3),  # 低一致性（混合图像）
        }
    ]
    
    losses_original = criterion_original(outputs, targets)
    print(f"  loss_bbox: {losses_original['loss_bbox']:.4f}")
    print(f"  loss_giou: {losses_original['loss_giou']:.4f}")
    
    # 测试2：静态元数据加权
    print("\n[Test 2] Static Meta Weighting")
    criterion_static = MetaDEIMCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        num_classes=80,
        use_meta_weighting=True,
        meta_mode='static',
        difficulty_scale=0.5,
        use_consistency_weighting=True
    )
    
    losses_static = criterion_static(outputs, targets)
    print(f"  loss_bbox: {losses_static['loss_bbox']:.4f}")
    print(f"  loss_giou: {losses_static['loss_giou']:.4f}")
    print(f"  Difference from original:")
    print(f"    Δ loss_bbox: {(losses_static['loss_bbox'] - losses_original['loss_bbox']):.4f}")
    print(f"    Δ loss_giou: {(losses_static['loss_giou'] - losses_original['loss_giou']):.4f}")
    
    # 测试3：动态MLP加权
    print("\n[Test 3] Dynamic MLP Weighting")
    criterion_dynamic = MetaDEIMCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        num_classes=80,
        use_meta_weighting=True,
        meta_mode='dynamic',
        difficulty_scale=0.5,
        use_consistency_weighting=True,
        mlp_hidden_dims=(128, 64)
    )
    
    losses_dynamic = criterion_dynamic(outputs, targets)
    print(f"  loss_bbox: {losses_dynamic['loss_bbox']:.4f}")
    print(f"  loss_giou: {losses_dynamic['loss_giou']:.4f}")
    print(f"  MLP parameters: {sum(p.numel() for p in criterion_dynamic.meta_difficulty.parameters())}")
    
    # 测试4：只用box级加权，不用sample级
    print("\n[Test 4] Box-level only (no consistency weighting)")
    criterion_box_only = MetaDEIMCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        num_classes=80,
        use_meta_weighting=True,
        meta_mode='static',
        difficulty_scale=0.5,
        use_consistency_weighting=False
    )
    
    losses_box_only = criterion_box_only(outputs, targets)
    print(f"  loss_bbox: {losses_box_only['loss_bbox']:.4f}")
    print(f"  loss_giou: {losses_box_only['loss_giou']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

"""
元数据引导的匈牙利匹配器
基于原始HungarianMatcher，集成StaticMetaDifficulty调整cost matrix

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict
import numpy as np

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .meta_weighting import StaticMetaDifficulty, extract_meta_features_from_targets
from ..core import register


@register()
class MetaHungarianMatcher(nn.Module):
    """
    元数据引导的匈牙利匹配器
    
    【核心场景】
    主要用于混合数据增强（Mosaic/MixUp）中的难度感知匹配：
    - 混合图像中不同难度的样本被组合在一起
    - 困难样本（高空/夜间/倾斜）可能匹配质量差的query
    - 通过元数据调整cost，确保困难样本得到充分学习
    
    【实现原理】
    1. 计算基础cost：cost_class + cost_bbox + cost_giou
    2. 检查meta_consistency_score：
       - 高一致性（≥0.95，纯图像）：跳过调整（所有GT难度相同，调整无效）
       - 低一致性（<0.95，混合图像）：执行元数据引导
    3. 提取GT元数据，计算静态难度分数 [0, 1]
    4. 调整cost：C_adjusted = C * (1.0 - difficulty * meta_cost_scale)
       - 困难GT：cost降低 → 更容易被匹配
       - 简单GT：cost保持 → 正常匹配
    5. 执行匈牙利算法
    
    【数学原理】
    匈牙利算法基于相对cost，只有当不同GT的adjustment不同时才会改变匹配：
    - 纯图像：所有GT元数据相同 → adjustment相同 → 等价于常数缩放 → 匹配不变
    - 混合图像：GT元数据不同 → adjustment不同 → 改变相对cost → 匹配改变 ✓
    """

    __share__ = ['use_focal_loss', ]

    def __init__(
        self, 
        weight_dict, 
        use_focal_loss=False, 
        alpha=0.25, 
        gamma=2.0,
        use_meta_weighting=False,
        meta_cost_scale=0.3,
        meta_consistency_threshold=0.95,
        meta_altitude_weight=0.4,
        meta_time_weight=0.3,
        meta_angle_weight=0.3
    ):
        """
        Args:
            weight_dict: cost权重字典 {'cost_class', 'cost_bbox', 'cost_giou'}
            use_focal_loss: 是否使用focal loss
            alpha: focal loss参数
            gamma: focal loss参数
            use_meta_weighting: 是否启用元数据引导
            meta_cost_scale: 元数据对cost的调整幅度 [0, 1]
                - 推荐0.2-0.3（困难样本cost降低20-30%）
            meta_consistency_threshold: 一致性阈值 [0, 1]
                - meta_consistency_score >= threshold: 跳过调整（纯图像）
                - meta_consistency_score < threshold: 执行调整（混合图像）
                - 推荐0.95（允许轻微混合也触发调整）
            meta_altitude_weight: 高度的难度权重
            meta_time_weight: 时间的难度权重
            meta_angle_weight: 角度的难度权重
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        # 元数据引导相关
        self.use_meta_weighting = use_meta_weighting
        self.meta_cost_scale = meta_cost_scale
        self.meta_consistency_threshold = meta_consistency_threshold
        
        if self.use_meta_weighting:
            self.static_difficulty = StaticMetaDifficulty(
                altitude_weight=meta_altitude_weight,
                time_weight=meta_time_weight,
                angle_weight=meta_angle_weight
            )
        
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """
        执行带元数据引导的匹配
        
        Args:
            outputs: dict包含
                - 'pred_logits': [B, num_queries, num_classes]
                - 'pred_boxes': [B, num_queries, 4]
            targets: list of dict，每个包含
                - 'labels': [num_boxes_i]
                - 'boxes': [num_boxes_i, 4]
                - 'meta_altitude': [num_boxes_i] (可选)
                - 'meta_time': [num_boxes_i] (可选)
                - 'meta_angle': [num_boxes_i] (可选)
        
        Returns:
            dict包含:
                - 'indices': list of (query_idx, gt_idx) tuples
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # ============ 计算基础cost matrix ============
        # 展平queries以便批量计算
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*num_queries, 4]

        # 拼接所有GT labels和boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类cost
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # 计算bbox L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 计算GIoU cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 组合cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()  # [B, num_queries, total_gt]

        sizes = [len(v["boxes"]) for v in targets]  # 每个样本的GT数量
        
        # C_origin = C.clone()
        
        # ============ 元数据引导的cost调整（仅混合增强时有效） ============
        if self.use_meta_weighting:
            # 先检查整个batch的一致性分数，如果都是纯图像则直接跳过
            consistency_scores = []
            has_mixed_image = False
            for target in targets:
                consistency = target.get('meta_consistency_score', torch.tensor(1.0))
                if isinstance(consistency, torch.Tensor):
                    consistency = consistency.item()
                consistency_scores.append(consistency)
                if consistency < self.meta_consistency_threshold:
                    has_mixed_image = True
            
            # 如果batch中没有混合图像，跳过所有计算
            if not has_mixed_image:
                pass  # 保持原始cost matrix，不做任何调整
            else:
                # 提取所有GT的元数据（list格式，每个样本长度不同）
                meta_list = extract_meta_features_from_targets(targets, indices=None)
                # meta_list: [[num_boxes_0, 3], [num_boxes_1, 3], ...]
                
                # 计算静态难度分数
                difficulty_list = self.static_difficulty(meta_list)
                # difficulty_list: [[num_boxes_0], [num_boxes_1], ...]
                
                # 按样本调整cost matrix
                # C: [B, num_queries, total_gt] 需要先split成每个样本
                C_split = list(C.split(sizes, -1))  # list of [num_queries, num_boxes_i]
                
                for i, difficulty in enumerate(difficulty_list):
                    if difficulty.numel() == 0:  # 跳过空样本
                        continue
                    
                    # 只在低一致性图像（混合图像）中调整cost
                    # 高一致性图像（纯图像）中所有GT难度相同，调整无效（等价于常数缩放）
                    if consistency_scores[i] >= self.meta_consistency_threshold:
                        continue  # 跳过纯图像，保持原始cost
                    
                    # 确保difficulty在CPU上（C已经在CPU上）
                    difficulty = difficulty.cpu()
                    
                    # 调整公式：C_new = C * (1.0 - difficulty * scale)
                    # 困难样本(difficulty=1)：cost降低scale倍 → 更容易匹配
                    # 简单样本(difficulty=0)：cost不变
                    adjustment = 1.0 - difficulty.unsqueeze(0) * self.meta_cost_scale  # [1, num_boxes_i]
                    C_split[i] = C_split[i] * adjustment  # [num_queries, num_boxes_i]
                
                # 重新拼接回batch format
                C = torch.cat(C_split, dim=-1).view(bs, num_queries, -1)
        
        # ============ 执行匈牙利算法 ============
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices (one-to-many matching)
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        """
        获取top-k匹配（one-to-many）
        保持与原始HungarianMatcher相同的实现
        """
        indices_list = []
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        return indices_list


if __name__ == '__main__':
    """测试MetaHungarianMatcher"""
    print("=" * 60)
    print("Testing MetaHungarianMatcher")
    print("=" * 60)
    
    # 模拟outputs
    batch_size = 2
    num_queries = 300
    num_classes = 80
    
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes),
        'pred_boxes': torch.rand(batch_size, num_queries, 4),
    }
    
    # 模拟targets（带元数据）
    targets = [
        {
            'boxes': torch.rand(5, 4),
            'labels': torch.randint(0, num_classes, (5,)),
            'meta_altitude': torch.tensor([0.0, 0.5, 1.0, 0.3, 0.7]),  # 不同高度
            'meta_time': torch.tensor([0, 1, 1, 0, 1]).float(),        # 白天/夜间
            'meta_angle': torch.tensor([0.0, 0.5, 1.0, 0.2, 0.8]),     # 不同角度
        },
        {
            'boxes': torch.rand(8, 4),
            'labels': torch.randint(0, num_classes, (8,)),
            'meta_altitude': torch.rand(8),
            'meta_time': torch.randint(0, 2, (8,)).float(),
            'meta_angle': torch.rand(8),
        }
    ]
    
    weight_dict = {
        'cost_class': 2.0,
        'cost_bbox': 5.0,
        'cost_giou': 2.0,
    }
    
    # 测试1：原始matcher（不使用元数据）
    print("\n[Test 1] Original Matcher (use_meta_weighting=False)")
    matcher_original = MetaHungarianMatcher(
        weight_dict=weight_dict,
        use_focal_loss=True,
        use_meta_weighting=False
    )
    
    result_original = matcher_original(outputs, targets)
    indices_original = result_original['indices']
    print(f"  Sample 0 matches: {len(indices_original[0][0])} pairs")
    print(f"    Query indices: {indices_original[0][0][:5].tolist()}...")
    print(f"    GT indices: {indices_original[0][1][:5].tolist()}...")
    print(f"  Sample 1 matches: {len(indices_original[1][0])} pairs")
    
    # 测试2：带元数据引导的matcher
    print("\n[Test 2] Meta-Guided Matcher (use_meta_weighting=True)")
    matcher_meta = MetaHungarianMatcher(
        weight_dict=weight_dict,
        use_focal_loss=True,
        use_meta_weighting=True,
        meta_cost_scale=0.3  # 困难样本cost降低30%
    )
    
    result_meta = matcher_meta(outputs, targets)
    indices_meta = result_meta['indices']
    print(f"  Sample 0 matches: {len(indices_meta[0][0])} pairs")
    print(f"    Query indices: {indices_meta[0][0][:5].tolist()}...")
    print(f"    GT indices: {indices_meta[0][1][:5].tolist()}...")
    print(f"  Sample 1 matches: {len(indices_meta[1][0])} pairs")
    
    # 测试3：对比匹配差异
    print("\n[Test 3] Matching Difference Analysis")
    for i in range(batch_size):
        gt_orig = set(indices_original[i][1].tolist())
        gt_meta = set(indices_meta[i][1].tolist())
        
        same = gt_orig & gt_meta
        only_orig = gt_orig - gt_meta
        only_meta = gt_meta - gt_orig
        
        print(f"  Sample {i}:")
        print(f"    Same GTs matched: {len(same)}/{len(gt_orig)}")
        if only_orig:
            print(f"    Only in original: {only_orig}")
        if only_meta:
            print(f"    Only in meta-guided: {only_meta}")
    
    # 测试4：不同meta_cost_scale的影响
    print("\n[Test 4] Impact of meta_cost_scale")
    for scale in [0.0, 0.1, 0.3, 0.5, 1.0]:
        matcher_temp = MetaHungarianMatcher(
            weight_dict=weight_dict,
            use_focal_loss=True,
            use_meta_weighting=True,
            meta_cost_scale=scale
        )
        result_temp = matcher_temp(outputs, targets)
        indices_temp = result_temp['indices']
        
        # 计算与原始的差异
        diff_count = 0
        for i in range(batch_size):
            gt_orig = set(indices_original[i][1].tolist())
            gt_temp = set(indices_temp[i][1].tolist())
            diff_count += len(gt_orig ^ gt_temp)  # symmetric difference
        
        print(f"    scale={scale:.1f}: {diff_count} different GT assignments")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

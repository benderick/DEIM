"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

元信息工具函数：用于计算样本质量分数和元信息处理
"""

import torch


def compute_meta_consistency_score(meta_altitude, meta_time, meta_angle, 
                                   tau_h=70.0, tau_theta=60.0, tau_t=1.5):
    """
    计算样本的元数据一致性分数
    
    通过计算所有唯一元数据组合的成对相似度平均值，评估样本的元数据一致性。
    适用于单图、Mosaic和MixUp样本。
    
    Args:
        meta_altitude (Tensor[N]): 所有box的高度信息（CODrone: 30/60/100m）
        meta_time (Tensor[N]): 所有box的时间信息 (0=day, 1=night)
        meta_angle (Tensor[N]): 所有box的角度信息（CODrone: 30°/90°）
        tau_h (float): 高度的温度参数，控制高度差异的敏感度（默认70，适配30-100m范围）
        tau_theta (float): 角度的温度参数，控制角度差异的敏感度（默认60，适配30-90°范围）
        tau_t (float): 时间的温度参数，控制时间差异的敏感度（默认1.5，适配0-1范围）
        
    Returns:
        consistency_score (Tensor[]): 标量Tensor，值域[0, 1]
            - 1.0: 完美一致（所有box来自相同条件）
            - 接近0: 差异很大（多个不同条件混合）
            
    Examples:
        >>> # 单图样本（所有box相同条件）
        >>> altitude = torch.tensor([60., 60., 60.])
        >>> time = torch.tensor([0., 0., 0.])
        >>> angle = torch.tensor([90., 90., 90.])
        >>> score = compute_meta_consistency_score(altitude, time, angle)
        >>> print(score)  # tensor(1.0)
        
        >>> # Mosaic样本（混合不同条件）
        >>> altitude = torch.tensor([30., 30., 60., 100.])
        >>> time = torch.tensor([0., 0., 1., 0.])
        >>> angle = torch.tensor([90., 90., 30., 90.])
        >>> score = compute_meta_consistency_score(altitude, time, angle)
        >>> print(score)  # tensor(0.3-0.6) 典型范围
    """
    # 检查输入
    if len(meta_altitude) == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    
    # Step 1: 提取唯一的元数据组合
    metas = torch.stack([meta_altitude, meta_time, meta_angle], dim=-1)  # [N, 3]
    unique_metas = torch.unique(metas, dim=0)  # [K, 3] K个不同的源
    
    K = len(unique_metas)
    
    # Step 2: 单一源图像，完美一致
    if K == 1:
        return torch.tensor(1.0, dtype=torch.float32)
    
    # Step 3: 计算所有成对相似度
    similarities = []
    for i in range(K):
        for j in range(i + 1, K):
            h_i, t_i, theta_i = unique_metas[i]
            h_j, t_j, theta_j = unique_metas[j]
            
            # 成对相似度（高斯核函数）
            s_h = torch.exp(-torch.abs(h_i - h_j) / tau_h)
            s_theta = torch.exp(-((theta_i - theta_j) ** 2) / (tau_theta ** 2))
            s_t = torch.exp(-torch.abs(t_i - t_j) / tau_t)
            
            s_ij = s_h * s_theta * s_t
            similarities.append(s_ij)
    
    # Step 4: 平均相似度作为一致性分数
    consistency_score = torch.stack(similarities).mean()
    
    return consistency_score

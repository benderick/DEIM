
"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
from PIL import Image
from .mosaic import Mosaic
from ...core import register

@register()
class MetaMosaic(Mosaic):
    """
    支持元数据拼接的 Mosaic 数据增强。
    继承自 Mosaic，增加了对 gt_altitude, gt_time, gt_angle 等实例级元数据的处理逻辑。
    """

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        """
        从缓存中创建 Mosaic 图像和标签。
        重写此方法以确保正确处理元数据 Tensor，并过滤掉无法拼接的字典类型元数据。
        """
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)

        mosaic_target = []
        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]

            merged_image.paste(img, placement_offsets[i])
            # 复制 target 以避免修改原始缓存
            new_target = target.copy()
            if 'boxes' in new_target:
                new_target['boxes'] = new_target['boxes'] + offsets[i]
            mosaic_target.append(new_target)

        return merged_image, self._merge_targets(mosaic_target)

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """
        从数据集中创建 Mosaic 图像和标签。
        """
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        
        # 预处理 targets，加上偏移量
        mosaic_targets = []
        for i, target in enumerate(targets):
            new_target = target.copy()
            if 'boxes' in new_target:
                new_target['boxes'] = new_target['boxes'] + offsets[i]
            mosaic_targets.append(new_target)

        return merged_image, self._merge_targets(mosaic_targets)

    def _merge_targets(self, targets):
        """
        合并多个 target 字典。
        自动过滤掉 'meta_info' 等非 Tensor 字段，防止 torch.cat 报错。
        """
        merged_target = {}
        # 以第一个 target 的键为基准
        keys = targets[0].keys()
        
        for key in keys:
            # 跳过图像级元信息字典，因为它无法拼接
            if key == 'meta_info':
                continue
                
            # 收集所有 target 的该字段值
            values = [t[key] for t in targets]
            
            # 如果是 Tensor，则进行拼接
            if isinstance(values[0], torch.Tensor):
                merged_target[key] = torch.cat(values, dim=0)
            else:
                # 对于非 Tensor 数据（如 image_id），通常保留列表或只取第一个
                # 这里为了兼容性，如果不是 Tensor 就不合并，或者根据需要处理
                # 在 DEIM 中，除了 boxes/labels/masks/gt_* 外，其他字段可能不需要拼接
                merged_target[key] = values

        return merged_target

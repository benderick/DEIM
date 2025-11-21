"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register
from .meta_utils import compute_meta_consistency_score


@register()
class MetaMosaic(T.Transform):
    """
    元信息引导的Mosaic增强。根据每张图像的元信息（altitude/time/angle）
    进行差异化的仿射变换，然后再拼接成Mosaic图像。
    
    增强策略：
    - 高海拔(100m)：更大的缩放变化（模拟远近感）
    - 低海拔(30m)：更大的旋转变化（模拟低空抖动）
    - 夜间(night)：更强的光度扰动
    - 倾斜角度(30°)：更大的平移变化（模拟视角偏移）
    """

    def __init__(self, output_size=320, max_size=None, rotation_range=10, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114, use_cache=True, max_cached_images=50,
                 random_pop=True, meta_guided=True, color_jitter_prob=0.5) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Base range of rotation in degrees (will be scaled by meta info).
            translation_range (tuple): Base range of translation (will be scaled by meta info).
            scaling_range (tuple): Base range of scaling factors (will be scaled by meta info).
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
            meta_guided (bool): Whether to use meta-information to guide augmentation.
            color_jitter_prob (float): Probability of applying color jitter for night images.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.base_rotation_range = rotation_range
        self.base_translation_range = translation_range
        self.base_scaling_range = scaling_range
        self.fill_value = fill_value
        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.meta_guided = meta_guided
        self.color_jitter_prob = color_jitter_prob
        
        # 预定义颜色抖动变换（用于夜间图像）
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def _extract_meta_info(self, target):
        """
        从target中提取元信息的平均值（因为一张图可能有多个boxes）
        
        Returns:
            dict: {'altitude': float, 'time': float, 'angle': float}
        """
        meta = {}
        
        # 提取altitude (默认60m)
        if 'meta_altitude' in target and len(target['meta_altitude']) > 0:
            meta['altitude'] = target['meta_altitude'][0].item()  # 取第一个box的值（同一图像应该相同）
        else:
            meta['altitude'] = 60.0
        
        # 提取time (0=day, 1=night, 默认day)
        if 'meta_time' in target and len(target['meta_time']) > 0:
            meta['time'] = target['meta_time'][0].item()
        else:
            meta['time'] = 0.0
        
        # 提取angle (默认90度)
        if 'meta_angle' in target and len(target['meta_angle']) > 0:
            meta['angle'] = target['meta_angle'][0].item()
        else:
            meta['angle'] = 90.0
        
        return meta

    def _get_meta_guided_affine_params(self, meta_info):
        """
        根据元信息计算差异化的仿射变换参数
        
        策略：
        - 高海拔(100m): scaling_factor *= 1.3 (更大缩放范围)
        - 低海拔(30m): rotation_factor *= 1.5 (更大旋转范围，模拟抖动)
        - 夜间: 需要颜色增强（在apply中处理）
        - 倾斜角度(30°): translation_factor *= 1.4 (更大平移)
        
        Args:
            meta_info (dict): {'altitude', 'time', 'angle'}
        
        Returns:
            dict: {'rotation_range', 'translation_range', 'scaling_range', 'need_color_jitter'}
        """
        altitude = meta_info['altitude']
        time_val = meta_info['time']
        angle = meta_info['angle']
        
        # 基础参数
        rotation_factor = 1.0
        translation_factor = 1.0
        scaling_factor = 1.0
        need_color_jitter = False
        
        # 根据海拔调整
        if altitude >= 100:
            scaling_factor = 1.3  # 高海拔增加缩放变化
            rotation_factor = 0.8  # 减少旋转（高空相对稳定）
        elif altitude <= 30:
            rotation_factor = 1.5  # 低海拔增加旋转（低空抖动）
            scaling_factor = 1.1
        else:  # 60m中等海拔
            rotation_factor = 1.0
            scaling_factor = 1.0
        
        # 根据时间调整
        if time_val > 0.5:  # night
            need_color_jitter = True
            translation_factor *= 1.2  # 夜间视觉定位不准，增加平移
        
        # 根据角度调整
        if angle <= 30:
            translation_factor *= 1.4  # 倾斜角度大，视角偏移大
            rotation_factor *= 1.2
        
        # 计算最终参数
        rotation_range = self.base_rotation_range * rotation_factor
        translation_range = tuple(t * translation_factor for t in self.base_translation_range)
        
        # scaling_range是(min, max)，需要围绕1.0对称调整
        base_min, base_max = self.base_scaling_range
        center = (base_min + base_max) / 2
        half_range = (base_max - base_min) / 2 * scaling_factor
        scaling_range = (max(0.1, center - half_range), center + half_range)
        
        return {
            'rotation_range': rotation_range,
            'translation_range': translation_range,
            'scaling_range': scaling_range,
            'need_color_jitter': need_color_jitter
        }

    def _apply_meta_guided_transform(self, image, target, meta_info):
        """
        对单张图像应用元信息引导的变换
        
        Args:
            image: PIL Image
            target: dict with boxes and meta fields
            meta_info: dict from _extract_meta_info
        
        Returns:
            (image, target): 变换后的图像和标签
        """
        if not self.meta_guided:
            # 如果不使用元信息引导，使用默认参数
            affine_transform = T.RandomAffine(
                degrees=self.base_rotation_range,
                translate=self.base_translation_range,
                scale=self.base_scaling_range,
                fill=self.fill_value
            )
            return affine_transform(image, target)
        
        # 获取元信息引导的参数
        params = self._get_meta_guided_affine_params(meta_info)
        
        # 应用颜色抖动（夜间图像）
        if params['need_color_jitter'] and random.random() < self.color_jitter_prob:
            image = self.color_jitter(image)
        
        # 应用仿射变换
        affine_transform = T.RandomAffine(
            degrees=params['rotation_range'],
            translate=params['translation_range'],
            scale=params['scaling_range'],
            fill=self.fill_value
        )
        
        return affine_transform(image, target)

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        # sample_indices = [1,2,3]
        for idx in sample_indices:
            # image, target = dataset.load_item(idx)
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"])) for idx in
                          sample_indices]  # sample 3 images
        mosaic_samples = [dict(img=image.copy(), labels=self._clone(target))] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        """
        从cache创建Mosaic，支持元信息引导的差异化增强
        
        流程：
        1. 对每张图提取元信息
        2. 根据元信息单独做仿射变换
        3. 将变换后的图像拼接成Mosaic
        """
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)

        mosaic_target = []
        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]
            
            # 提取元信息并应用差异化增强
            meta_info = self._extract_meta_info(target)
            img, target = self._apply_meta_guided_transform(img, target, meta_info)

            merged_image.paste(img, placement_offsets[i])
            target['boxes'] = target['boxes'] + offsets[i]
            mosaic_target.append(target)

        merged_target = {}
        for key in mosaic_target[0]:
            if key == 'meta_consistency_score':
                # Image级元信息，不拼接，稍后重新计算
                continue
            merged_target[key] = torch.cat([target[key] for target in mosaic_target])

        # 计算Mosaic样本的元数据一致性分数
        if 'meta_altitude' in merged_target:
            merged_target['meta_consistency_score'] = compute_meta_consistency_score(
                merged_target['meta_altitude'],
                merged_target['meta_time'],
                merged_target['meta_angle']
            )
        else:
            # 没有元信息时使用默认值
            merged_target['meta_consistency_score'] = torch.tensor(1.0, dtype=torch.float32)

        return merged_image, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """
        从dataset创建Mosaic，支持元信息引导的差异化增强
        
        流程：
        1. 对每张图提取元信息
        2. 根据元信息单独做仿射变换
        3. 将变换后的图像拼接成Mosaic
        """
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        
        transformed_images = []
        transformed_targets = []
        
        # 对每张图像单独应用元信息引导的变换
        for i, (img, target) in enumerate(zip(images, targets)):
            meta_info = self._extract_meta_info(target)
            img, target = self._apply_meta_guided_transform(img, target, meta_info)
            transformed_images.append(img)
            transformed_targets.append(target)
        
        # 拼接变换后的图像
        for i, img in enumerate(transformed_images):
            merged_image.paste(img, placement_offsets[i])

        # 合并targets
        merged_target = {}
        for key in transformed_targets[0]:
            if key == 'boxes':
                values = [target[key] + offsets[i] for i, target in enumerate(transformed_targets)]
            elif key == 'meta_consistency_score':
                # Image级元信息，不拼接，稍后重新计算
                continue
            else:
                values = [target[key] for target in transformed_targets]

            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        # 计算Mosaic样本的元数据一致性分数
        if 'meta_altitude' in merged_target:
            merged_target['meta_consistency_score'] = compute_meta_consistency_score(
                merged_target['meta_altitude'],
                merged_target['meta_time'],
                merged_target['meta_angle']
            )
        else:
            # 没有元信息时使用默认值
            merged_target['meta_consistency_score'] = torch.tensor(1.0, dtype=torch.float32)

        return merged_image, merged_target

    @staticmethod
    def _clone(tensor_dict):
        return {key: value.clone() for (key, value) in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        # Clamp boxes and convert target formats
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(mosaic_target['boxes'], 'boxes', box_format='xyxy',
                                                          spatial_size=mosaic_image.size[::-1])
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')

        # 注意：仿射变换已经在 _apply_meta_guided_transform 中对每张图单独应用了
        # 这里不再统一应用，保持拼接后的结果

        return mosaic_image, mosaic_target, dataset

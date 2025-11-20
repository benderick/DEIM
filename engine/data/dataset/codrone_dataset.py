
import torch
import torchvision
from .coco_dataset import CocoDetection, ConvertCocoPolysToMask, mscoco_category2label
from ...core import register
from typing import Dict

def parse_meta_from_filename(filename: str) -> Dict:
        """
        从文件名解析元信息
        
        格式: {location}_{time}_{altitude}_{angle}_frame_{frame_num}.jpg
        例如: chenhuachengpark_day_30m_30c_frame_750.jpg
        
        Returns:
            meta: {
                'time': 'day' or 'night',
                'altitude': 30/60/100,
                'angle': 30/90,
                'location': str,
                'frame': int,
                'filename': str
            }
        """
        # 去掉扩展名
        name = filename.replace('.jpg', '').replace('.png', '')
        parts = name.split('_')
        
        meta = {
            'time': None,
            'altitude': None,
            'angle': None,
            'location': None,
            'frame': None,
            'filename': filename
        }
        
        # 解析逻辑
        location_parts = []
        for i, part in enumerate(parts):
            if part in ['day', 'night']:
                meta['time'] = part
                meta['location'] = '_'.join(location_parts) if location_parts else 'unknown'
            elif part.endswith('m') and len(part) > 1:
                try:
                    meta['altitude'] = int(part[:-1])
                except:
                    pass
            elif part.endswith('c') and len(part) > 1:
                try:
                    meta['angle'] = int(part[:-1])
                except:
                    pass
            elif part == 'frame' and i + 1 < len(parts):
                try:
                    meta['frame'] = int(parts[i + 1])
                except:
                    pass
            else:
                # 收集地点名称的部分
                if meta['time'] is None:  # 时间信息之前的都是地点
                    location_parts.append(part)
        
        return meta

from .._misc import convert_to_tv_tensor

@register()
class CODroneDetection(CocoDetection):
    def load_item(self, idx):
        # 1. 调用 super 的 __getitem__ 获取原始数据 (image, target_list)
        # 注意：这里不能调用 super().load_item(idx)，因为那是 CocoDetection 的实现，会直接返回处理好的数据
        # 我们需要从头开始处理，以便插入 meta_info
        
        # 调用 torchvision.datasets.CocoDetection 的 __getitem__
        # 由于 CocoDetection 继承自 torchvision.datasets.CocoDetection
        # 我们使用 super(CocoDetection, self) 指向 torchvision 类
        # 但 CocoDetection 定义在当前模块，继承关系是 CocoDetection -> torchvision.datasets.CocoDetection
        # 所以 super(CocoDetection, self) 是对的。
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # 2. 调用 self.prepare (ConvertCocoPolysToMask)
        # 这会将 boxes 转为 xyxy 格式，并处理 labels 等
        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        # 3. 解析并注入 meta_info
        # 必须在 prepare 之后，因为 prepare 会重构 target 字典
        coco = self.coco
        img_info = coco.loadImgs(image_id)[0]
        file_name = img_info['file_name']
        meta_info = parse_meta_from_filename(file_name)
        
        # --- META-GUIDED MODIFICATION START ---
        # 将 meta_info 转换为 Instance-level 的 Tensor
        # 这样在 Mosaic/Mixup 时，这些 Tensor 会跟随 boxes 一起被拼接
        num_boxes = len(target['boxes'])
        
        # Altitude (30, 60, 100) -> Tensor
        alt = meta_info.get('altitude', 30)
        if alt is None: alt = 30
        target['gt_altitude'] = torch.full((num_boxes,), alt, dtype=torch.float32)
        
        # Time (day=0, night=1) -> Tensor
        time_val = meta_info.get('time', 'day')
        time_idx = 1.0 if time_val == 'night' else 0.0
        target['gt_time'] = torch.full((num_boxes,), time_idx, dtype=torch.float32)
        
        # Angle (90, 30) -> Tensor
        angle = meta_info.get('angle', 90)
        if angle is None: angle = 90
        target['gt_angle'] = torch.full((num_boxes,), angle, dtype=torch.float32)
        
        # 保留原始 meta_info 用于调试或非增强场景
        # target['meta_info'] = meta_info
        # --- META-GUIDED MODIFICATION END ---

        target['idx'] = torch.tensor([idx])

        # 4. 转换为 TV Tensor (保持与 CocoDetection 一致)
        # 注意：这里不进行归一化，也不转 cxcywh，这些由后续的 transforms 处理
        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

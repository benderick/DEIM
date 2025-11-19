
import torch
import sys
import os
from rich import print

# 添加项目根目录到 sys.path，以便能导入 engine 模块
sys.path.append(os.getcwd())

from engine.data.dataset.codrone_dataset import CODroneDetection
from engine.data.dataset.coco_dataset  import CocoDetection
import torchvision.transforms as T

def test_codrone_dataset():
    # 路径配置
    img_folder = "data/CODrone/train/images"
    ann_file = "data/CODrone/train/annotations/instances_default.json"

    print(f"Testing CODroneDetection with:")
    print(f"  Image folder: {img_folder}")
    print(f"  Annotation file: {ann_file}")

    # 定义一个简单的 transform，这里我们不需要复杂的增强，只需要能运行即可
    # CODroneDetection 内部会调用 transforms，如果传入 None 可能会报错或者不进行任何处理
    # 这里我们传入一个简单的 Compose，不做任何操作，或者只做 ToTensor (虽然 dataset 内部可能已经处理了)
    # 注意：CODroneDetection 继承自 CocoDetection，它的 __getitem__ 会调用 transforms
    # 我们定义一个简单的伪 transform
    class SimpleTransform:
        def __call__(self, image, target, dataset=None):
            return image, target, None

    dataset = CODroneDetection(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=SimpleTransform(),
        return_masks=False
    )
    
    # dataset = CocoDetection(
    #     img_folder=img_folder,
    #     ann_file=ann_file,
    #     transforms=SimpleTransform(),
    #     return_masks=False
    # )

    print(f"Dataset categories: {dataset.categories}")
    print(f"Dataset length: {len(dataset)}")

    # 随机抽取几个样本进行检查
    indices = [0, 10, 50]
    indices = [0]
    for idx in indices:
        if idx >= len(dataset):
            continue
            
        print(f"\n--- Checking sample at index {idx} ---")
        image, target = dataset[idx]
        
        print(f"Image size: {image.size}, Image type: {type(image)}")  # PIL Image 的 size 属性是 (width, height)
        
        print(target)

        # 检查 meta_info
        if 'meta_info' in target:
            print(f"Meta Info: {target['meta_info']}")
        else:
            print("ERROR: 'meta_info' not found in target!")

        # 检查 boxes
        if 'boxes' in target:
            boxes = target['boxes']
            print(f"Boxes shape: {boxes.shape}")
            if len(boxes) > 0:
                print(f"First box (cx, cy, w, h): {boxes[0]}")
                # 检查是否归一化 (0-1之间)
                is_normalized = (boxes >= 0).all() and (boxes <= 1).all()
                print(f"Boxes normalized? {is_normalized}")
        else:
            print("No boxes in this sample.")
            
        # 检查 image_id
        print(f"Image ID: {target['image_id']}")

if __name__ == "__main__":
    test_codrone_dataset()

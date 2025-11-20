
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import random
from collections import Counter


# Add path to engine
sys.path.append(os.getcwd())

from engine.data.transforms.mosaic import Mosaic
from engine.data.transforms import RandomZoomOut
from engine.data.transforms import RandomIoUCrop, RandomHorizontalFlip, Resize, ConvertPILImage, ConvertBoxes, RandomPhotometricDistort
from engine.data.transforms.meta_mosaic import MetaMosaic
from engine.data.dataloader import BatchImageCollateFunction
from engine.data.dataset.codrone_dataset import CODroneDetection
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms.v2 import SanitizeBoundingBoxes
from engine.data.transforms.meta_transforms import MetaSanitizeBoundingBoxes

# Ensure output directory exists
os.makedirs("visualization_results", exist_ok=True)

class SimpleCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, dataset):
        for t in self.transforms:
            # Mosaic expects (image, target, dataset)
            # We can try passing all 3, if it fails, pass 2?
            # But Mosaic.forward(*inputs) handles it.
            # Let's assume all transforms here are compatible or we wrap them.
            if isinstance(t, MetaMosaic) or isinstance(t, Mosaic):
                image, target, dataset = t(image, target, dataset)
            else:
                # For other transforms, we might need to adapt.
                # But for this visualization, we only use MetaMosaic in the dataset pipeline.
                image, target = t(image, target)
        return image, target, dataset

class DummyTransform:
    def __call__(self, image, target, dataset=None):
        return image, target, None

def draw_boxes(image, target, filename):
    draw = ImageDraw.Draw(image)
    
    if 'boxes' not in target or len(target['boxes']) == 0:
        print(f"No boxes to draw for {filename}")
        image.save(f"visualization_results/{filename}")
        return

    boxes = target['boxes']
    
    # Check if boxes are normalized
    is_normalized = (boxes.max() <= 1.0)
    w, h = image.size
    
    for i, box in enumerate(boxes):
        if is_normalized:
            x1 = box[0] * w
            y1 = box[1] * h
            x2 = box[2] * w
            y2 = box[3] * h
        else:
            x1, y1, x2, y2 = box.tolist()
            
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Text info
        info_text = []
        if 'gt_altitude' in target:
            alt = target['gt_altitude'][i].item()
            info_text.append(f"Alt:{alt:.0f}")
        if 'gt_time' in target:
            time_val = "Night" if target['gt_time'][i].item() > 0.5 else "Day"
            info_text.append(f"{time_val}")
        if 'gt_angle' in target:
            angle = target['gt_angle'][i].item()
            info_text.append(f"Ang:{angle:.0f}")
            
        text = ", ".join(info_text)
        if text:
            draw.text((x1, y1), text, fill="yellow")
            # draw.text((x1-1, y1-1), text, fill="black") # Stroke effect

    image.save(f"visualization_results/{filename}")
    print(f"Saved {filename}")

def print_stats(target, prefix=""):
    print(f"\n[{prefix}] Statistics:")
    if 'boxes' not in target:
        print("  No boxes found.")
        return

    num_boxes = len(target['boxes'])
    print(f"  Total Objects: {num_boxes}")
    
    meta_keys = ['gt_altitude', 'gt_time', 'gt_angle']
    for key in meta_keys:
        if key in target:
            data = target[key]
            print(f"  {key}: Count={len(data)}")
            if len(data) != num_boxes:
                print(f"  WARNING: {key} count ({len(data)}) does not match boxes count ({num_boxes})!")
            
            # Value distribution
            vals = data.tolist()
            # Round floats for cleaner counting
            vals = [round(v, 1) for v in vals]
            counts = Counter(vals)
            print(f"    Distribution: {dict(counts)}")
        else:
            print(f"  {key}: Not found in target")

def visualize():
    print("Initializing CODrone Dataset...")
    img_folder = "data/CODrone/train/images"
    ann_file = "data/CODrone/train/annotations/instances_default.json"
    
    # 1. Visualize Mosaic (Integrated into Dataset)
    print("\n=== Visualizing Mosaic (via Dataset Pipeline) ===")
    
    mosaic_op = MetaMosaic(
        output_size=640, 
        rotation_range=10,
        translation_range=[0.1, 0.1],
        scaling_range=[0.5, 1.5],
        probability=1.0,
        fill_value=114, 
        use_cache=False
    )
    
    # mosaic_op = MetaMosaic(
    #     output_size=640, 
    #     rotation_range=0,
    #     translation_range=[0,0],
    #     scaling_range=[1,1],
    #     probability=1.0,
    #     fill_value=114, 
    #     use_cache=False
    # )
    
    sanitize_op = MetaSanitizeBoundingBoxes(min_size=1.0, custom_fields=['gt_altitude', 'gt_time', 'gt_angle'])
    
    sanitize_op1 = MetaSanitizeBoundingBoxes(min_size=1.0, custom_fields=['gt_altitude', 'gt_time', 'gt_angle'])
    
    random_zoom_out_op = RandomZoomOut(fill=0)
    
    random_iou_crop_op = RandomIoUCrop(p=1.0)
    
    random_photometric_distort = RandomPhotometricDistort(p=1.0)
    
    random_horizontal_flip = RandomHorizontalFlip(p=1.0)
    
    resize = Resize(size=[640,640])
    
    convert_img = ConvertPILImage()
    convert_boxes = ConvertBoxes(fmt="cxcywh", normalize=True)
    
    
    # Use SimpleCompose to wrap Mosaic
    transforms = SimpleCompose([mosaic_op, random_photometric_distort, random_zoom_out_op, random_iou_crop_op,sanitize_op, random_horizontal_flip, resize, sanitize_op1])
    
    dataset_mosaic = CODroneDetection(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=transforms,
        return_masks=False
    )
    
    print(f"Dataset loaded with {len(dataset_mosaic)} images.")
    
    # Pick a random index
    idx = random.randint(0, len(dataset_mosaic) - 1)
    print(f"Loading sample index: {idx}")
    
    # This calls __getitem__, which calls transforms (Mosaic)
    img_mosaic, target_mosaic = dataset_mosaic[idx]
    
    print("hello")
    
    if isinstance(img_mosaic, np.ndarray):
        img_mosaic = Image.fromarray(img_mosaic)
    
    print_stats(target_mosaic, prefix="Mosaic")
    
    draw_boxes(img_mosaic, target_mosaic, "real_mosaic_result.jpg")
    
    
    # # 2. Visualize Mixup (via Collate Function)
    # print("\n=== Visualizing Mixup (via Collate Function) ===")
    
    # # For Mixup, we need a dataset that returns tensors, but WITHOUT Mosaic (to isolate Mixup effect)
    # # Or we can use the same dataset if we want Mosaic + Mixup.
    # # Let's use a clean dataset for Mixup visualization to be clear.
    
    # class ToTensorTransform:
    #     def __call__(self, image, target):
    #         return to_tensor(image), target

    # transforms_mixup = SimpleCompose([ToTensorTransform()])
    
    # dataset_mixup = CODroneDetection(
    #     img_folder=img_folder,
    #     ann_file=ann_file,
    #     transforms=transforms_mixup,
    #     return_masks=False
    # )
    
    # collate_fn = BatchImageCollateFunction(
    #     mixup_prob=1.0,
    #     mixup_epochs=[0, 100],
    #     base_size=640
    # )
    # collate_fn.set_epoch(10)
    
    # # Pick 2 random images
    # idx1 = random.randint(0, len(dataset_mixup) - 1)
    # idx2 = random.randint(0, len(dataset_mixup) - 1)
    
    # # We need to manually construct the batch list as DataLoader would
    # # dataset[i] returns (img_tensor, target)
    # item1 = dataset_mixup[idx1]
    # item2 = dataset_mixup[idx2]
    
    # # Resize manually because BatchImageCollateFunction expects somewhat consistent sizes or handles it?
    # # Actually, BatchImageCollateFunction handles resizing if base_size is set.
    # # But we need to make sure 'area' is present and correct if we resized.
    # # Since we didn't resize in transform, we rely on collate_fn.
    
    # # However, collate_fn expects a list of (image, target).
    # batch = [item1, item2]
    
    # out_imgs, out_tgts = collate_fn(batch)
    
    # # Visualize first image (mixed)
    # mix_img = to_pil_image(out_imgs[0])
    # mix_tgt = out_tgts[0]
    
    # # Denormalize boxes if needed (Mixup usually keeps them absolute if input is absolute, 
    # # but let's check if ToTensor normalized them? No, ToTensor only normalizes Image. 
    # # Boxes in CODroneDetection are converted to TVTensor but not normalized unless we added ConvertBoxes transform)
    
    # print_stats(mix_tgt, prefix="Mixup")
    # draw_boxes(mix_img, mix_tgt, "real_mixup_result.jpg")
    
    print("\nVisualization Complete.")

if __name__ == "__main__":
    visualize()

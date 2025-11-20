from typing import Any, Dict, cast
import torch
from torchvision.transforms.v2 import SanitizeBoundingBoxes
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.transforms.v2._utils import get_bounding_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from ...core import register

@register()
class MetaSanitizeBoundingBoxes(SanitizeBoundingBoxes):
    def __init__(self, custom_fields=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_fields = custom_fields or []

    def forward(self, *inputs: Any) -> Any:
        assert len(inputs) >= 1
        inputs = inputs if len(inputs) > 1 else inputs[0]

        labels = self._labels_getter(inputs)
        if labels is not None and not isinstance(labels, torch.Tensor):
            raise ValueError(
                f"The labels in the input to forward() must be a tensor or None, got {type(labels)} instead."
            )

        flat_inputs, spec = tree_flatten(inputs)
        boxes = get_bounding_boxes(flat_inputs)

        if labels is not None and boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of boxes (shape={boxes.shape}) and number of labels (shape={labels.shape}) do not match."
            )
        
        # add custom fields 添加自定义字段
        customs = []
        if len(self.custom_fields) > 0 and isinstance(inputs, (tuple, list)) and isinstance(inputs[1], dict):   
            for field in self.custom_fields:
                if field in inputs[1]:
                    assert boxes.shape[0] == inputs[1][field].shape[0], \
                        f"Number of boxes ({boxes.shape[0]}) and number of entries in custom field '{field}' ({inputs[1][field].shape[0]}) do not match."
                    customs.append(inputs[1][field])
                else:
                    raise ValueError(f"Custom field '{field}' not found in target dictionary.")

        boxes = cast(
            tv_tensors.BoundingBoxes,
            F.convert_bounding_box_format(
                boxes,
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            ),
        )
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        valid = (ws >= self.min_size) & (hs >= self.min_size) & (boxes >= 0).all(dim=-1)
        # TODO: Do we really need to check for out of bounds here? All
        # transforms should be clamping anyway, so this should never happen?
        image_h, image_w = boxes.canvas_size
        valid &= (boxes[:, 0] <= image_w) & (boxes[:, 2] <= image_w)
        valid &= (boxes[:, 1] <= image_h) & (boxes[:, 3] <= image_h)

        params = dict(valid=valid.as_subclass(torch.Tensor), labels=labels, customs=customs)
        flat_outputs = [
            # Even-though it may look like we're transforming all inputs, we don't:
            # _transform() will only care about BoundingBoxeses and the labels
            self._transform(inpt, params)
            for inpt in flat_inputs
        ]

        return tree_unflatten(flat_outputs, spec)
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            is_label = inpt is not None and inpt is params["labels"]
            is_custom = inpt is not None and any(inpt is custom for custom in params["customs"])
            is_bounding_boxes_or_mask = isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask))

            if not (is_label or is_custom or is_bounding_boxes_or_mask):
                return inpt

            output = inpt[params["valid"]]

            if is_label or is_custom:
                return output

            return tv_tensors.wrap(output, like=inpt)
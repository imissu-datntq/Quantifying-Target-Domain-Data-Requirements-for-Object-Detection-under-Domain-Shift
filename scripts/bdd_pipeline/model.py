"""One fixed native-resolution detector contract across all budgets."""
from __future__ import annotations

import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from .data import CLASSES, HEIGHT, WIDTH

MODEL_CONTRACT = {
    "architecture": "fasterrcnn_resnet50_fpn",
    "class_names": list(CLASSES), "background_id": 0,
    "input_width": WIDTH, "input_height": HEIGHT,
    "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
    "min_size": HEIGHT, "max_size": WIDTH, "size_divisible": 32,
    "roi_score_threshold": 0.001, "roi_nms_threshold": 0.5,
    "detections_per_image": 100, "rpn_pre_nms_train": 2000,
    "rpn_post_nms_train": 2000, "rpn_pre_nms_eval": 1000, "rpn_post_nms_eval": 1000,
    "trainable_backbone": "layer2,layer3,layer4; FPN/RPN/ROI trainable",
    "normalization_layers": "FrozenBatchNorm2d",
    "frozen_batch_norm_epsilon": 0.0,
}


def freeze_batch_norm(module: torch.nn.Module) -> None:
    """Match COCO-pretrained FrozenBN structure even for offline random smoke."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            replacement = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            replacement.weight.copy_(child.weight.detach())
            replacement.bias.copy_(child.bias.detach())
            replacement.running_mean.copy_(child.running_mean.detach())
            replacement.running_var.copy_(child.running_var.detach())
            setattr(module, name, replacement)
        else:
            freeze_batch_norm(child)


def build_model(initialization: str = "coco") -> torch.nn.Module:
    if initialization not in {"coco", "imagenet", "none"}:
        raise ValueError("Initialization must be coco, imagenet or none")
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if initialization == "coco" else None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if initialization == "imagenet" else None,
        min_size=HEIGHT, max_size=WIDTH, image_mean=MODEL_CONTRACT["image_mean"],
        image_std=MODEL_CONTRACT["image_std"], box_score_thresh=0.001,
        box_nms_thresh=0.5, box_detections_per_img=100,
    )
    freeze_batch_norm(model.backbone)
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = MODEL_CONTRACT["frozen_batch_norm_epsilon"]
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad_(name.startswith(("layer2.", "layer3.", "layer4.")))
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1)
    return model

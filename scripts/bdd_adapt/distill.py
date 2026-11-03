"""ALDI++ soft distillation on torchvision FasterRCNN internals.

Teacher (EMA, eval mode, no grad) sees the weak view; the student sees the strong view of the same
image, so the geometry is identical. Hard pseudo-labels (teacher detections >= threshold) are used
only to sample anchors and RoI proposals; the losses are soft (ALDI-Best config):
  RPN objectness  BCE(student logit, sigmoid(teacher logit))   on sampled anchors
  RPN regression  smooth-L1(beta 0, mean) student vs teacher deltas on pseudo-foreground anchors
  RoI class       CE(student logits, softmax(teacher logits))  on sampled proposals
  RoI regression  smooth-L1(beta 0, sum / proposals) for the teacher's argmax class where the
                  teacher's argmax is not background
Both models run their RoI heads on the student's sampled proposals.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models.detection.rpn import concat_box_prediction_layers

THRESHOLD = 0.8


def pseudo_labels(teacher, images_t, features_t, threshold: float = THRESHOLD) -> list[dict]:
    """Teacher detections (eval-mode post-processing) above the threshold, in the model-input frame."""
    proposals, _ = teacher.rpn(images_t, features_t)
    detections, _ = teacher.roi_heads(features_t, proposals, images_t.image_sizes)
    labels = []
    for detection in detections:
        keep = detection["scores"] >= threshold
        labels.append({"boxes": detection["boxes"][keep].detach().float(), "labels": detection["labels"][keep].detach()})
    return labels


def _rpn_outputs(model, images, features: list):
    objectness, deltas = model.rpn.head(features)
    shapes = [o[0].shape for o in objectness]
    per_level = [s[0] * s[1] * s[2] for s in shapes]
    objectness, deltas = concat_box_prediction_layers(objectness, deltas)
    return objectness.flatten(), deltas, per_level


def _roi_outputs(model, features: dict, proposals: list, image_sizes):
    pooled = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
    return model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))


def distillation_losses(student, teacher, weak: list[torch.Tensor], strong: list[torch.Tensor],
                        threshold: float = THRESHOLD) -> tuple[dict, dict]:
    """Soft distillation losses (student graph attached) and diagnostics. Student must be in train mode."""
    if not student.training or teacher.training:
        raise ValueError("Student must be in train mode and the teacher in eval mode")
    with torch.no_grad():
        images_t, _ = teacher.transform(weak)
        features_t = teacher.backbone(images_t.tensors)
        targets = pseudo_labels(teacher, images_t, features_t, threshold)
        objectness_t, deltas_t, _ = _rpn_outputs(teacher, images_t, list(features_t.values()))
    images_s, _ = student.transform(strong)
    if images_s.tensors.shape != images_t.tensors.shape or images_s.image_sizes != images_t.image_sizes:
        raise ValueError("Weak and strong views must share geometry")
    features_s = student.backbone(images_s.tensors)
    levels = list(features_s.values())
    objectness_s, deltas_s, per_level = _rpn_outputs(student, images_s, levels)
    anchors = student.rpn.anchor_generator(images_s, levels)
    count = len(anchors)
    decoded = student.rpn.box_coder.decode(deltas_s.detach(), anchors).view(count, -1, 4)
    proposals, _ = student.rpn.filter_proposals(decoded, objectness_s.detach().view(-1, 1), images_s.image_sizes, per_level)

    # RPN: sample anchors against pseudo boxes exactly as the supervised RPN loss would
    anchor_labels, _ = student.rpn.assign_targets_to_anchors(anchors, targets)
    positive, negative = student.rpn.fg_bg_sampler(anchor_labels)
    positive = torch.where(torch.cat(positive))[0]
    sampled = torch.cat([positive, torch.where(torch.cat(negative))[0]])
    losses = {"loss_obj_bce": F.binary_cross_entropy_with_logits(
        objectness_s[sampled].float(), torch.sigmoid(objectness_t[sampled].float()))}
    losses["loss_rpn_l1"] = (F.smooth_l1_loss(deltas_s[positive].float(), deltas_t[positive].float(), beta=0.0)
                             if positive.numel() else deltas_s.sum() * 0.0)

    # RoI: both heads on the student's sampled proposals (pseudo boxes appended, as in training)
    sampled_proposals, _, _, _ = student.roi_heads.select_training_samples(proposals, targets)
    sampled_proposals = [p.detach() for p in sampled_proposals]
    logits_s, regression_s = _roi_outputs(student, features_s, sampled_proposals, images_s.image_sizes)
    with torch.no_grad():
        logits_t, regression_t = _roi_outputs(teacher, features_t, sampled_proposals, images_t.image_sizes)
    logits_s, logits_t = logits_s.float(), logits_t.float()
    losses["loss_cls_ce"] = -(F.softmax(logits_t, dim=1) * F.log_softmax(logits_s, dim=1)).sum(1).mean()
    classes = logits_t.shape[1]
    foreground_class = logits_t.argmax(1)
    foreground = foreground_class != 0  # torchvision: background is class 0
    rows = torch.where(foreground)[0]
    student_boxes = regression_s.float().view(-1, classes, 4)[rows, foreground_class[rows]]
    teacher_boxes = regression_t.float().view(-1, classes, 4)[rows, foreground_class[rows]]
    losses["loss_roih_l1"] = F.smooth_l1_loss(student_boxes, teacher_boxes, beta=0.0, reduction="sum") / max(1, logits_t.shape[0])
    diagnostics = {"pseudo_boxes": sum(len(t["boxes"]) for t in targets), "rpn_positive": int(positive.numel()),
                   "roi_teacher_foreground": int(rows.numel()), "roi_proposals": int(logits_t.shape[0])}
    return losses, diagnostics


@torch.no_grad()
def ema_update(teacher, student, alpha: float) -> None:
    """teacher <- alpha * teacher + (1 - alpha) * student, over parameters and float buffers."""
    teacher_state, student_state = teacher.state_dict(), student.state_dict()
    if teacher_state.keys() != student_state.keys():
        raise ValueError("Teacher/student structure differs")
    for key, value in teacher_state.items():
        source = student_state[key]
        if value.dtype.is_floating_point:
            value.mul_(alpha).add_(source.detach(), alpha=1 - alpha)
        else:
            value.copy_(source)

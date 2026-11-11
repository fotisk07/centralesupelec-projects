from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from challenge.dataset import Sample, class_names, yolo_to_xyxy_torch


def _greedy_match_ious(ious: torch.Tensor) -> list[float]:
    if ious.numel() == 0:
        return []

    work = ious.clone()
    n_gt, n_pr = work.shape
    n_matches = min(n_gt, n_pr)
    matched: list[float] = []

    for _ in range(n_matches):
        flat_idx = torch.argmax(work)
        best = work.view(-1)[flat_idx].item()
        if best <= 0.0:
            break

        r = int(flat_idx // n_pr)
        c = int(flat_idx % n_pr)
        matched.append(best)

        # Remove row/column so each GT/pred can be used at most once.
        work[r, :] = -1.0
        work[:, c] = -1.0

    return matched


### The dataclass approach might seem overkill but it is a very simple way
### to hide the details from main and show them in a pretty way just by running print
### Even thought the kaggle metric does not give class info, having them here
### is very informative because we can tune the filters to improve our scores
@dataclass
class FrameStats:
    score: float  # The official metric score for this frame
    gt_count: int
    pred_count: int

    def __str__(self):
        return f"  score={self.score:.3f}  gt={self.gt_count}  pred={self.pred_count}"


@dataclass
class ClassStats:
    mean_iou: float  # Average greedy-matched IoU across all GT boxes of this class
    recall: float  # Fraction of GT boxes with matched IoU > iou_threshold
    precision: float  # Fraction of pred boxes with matched IoU > iou_threshold

    def __str__(self):
        return f"  mIoU={self.mean_iou:.3f}  recall={self.recall:.3f}  precision={self.precision:.3f}"


@dataclass
class EvalStats:
    score: float

    per_class: dict[int, ClassStats]

    n_frames: int
    n_frames_both_empty: int
    n_frames_no_pred: int

    def __str__(self):
        lines = [
            f"{'─' * 45}",
            f"  Score : {self.score:.4f}",
            f"  Frames: {self.n_frames}  trivial={self.n_frames_both_empty}  no_pred={self.n_frames_no_pred}",
            f"{'─' * 45}",
            f"  {'class':<10} {'mIoU':>6}  {'recall':>7}  {'precision':>10}",
            f"  {'─' * 38}",
        ]
        for cls_id, cs in sorted(self.per_class.items()):
            lines.append(
                f"  {class_names[cls_id]:<10} {cs.mean_iou:>6.3f}  {cs.recall:>7.3f}  {cs.precision:>10.3f}"
            )
        lines.append(f"{'─' * 45}")
        return "\n".join(lines)


def evaluate_model_with_stats(
    pred_boxes: list[np.ndarray],
    loader: Iterable[Sample],
    iou_threshold: float = 0.3,
) -> EvalStats:
    frame_scores = []
    n_frames_both_empty = 0
    n_frames_no_pred = 0

    # Per-class accumulators
    class_iou_sum: dict[int, float] = defaultdict(float)
    class_gt_count: dict[int, int] = defaultdict(int)
    class_pred_count: dict[int, int] = defaultdict(int)
    class_tp: dict[int, int] = defaultdict(int)  # matched IoU > iou_threshold

    for sample, pred in tqdm(zip(loader, pred_boxes), total=len(pred_boxes)):
        img = sample["image"]
        gt = sample["targets"]
        H, W = img.shape[:2]
        Ng, Np = len(gt), len(pred)

        # Frame score (exact metric)
        if Ng == 0 and Np == 0:
            frame_scores.append(1.0)
            n_frames_both_empty += 1
            continue

        if Ng == 0 or Np == 0:
            frame_scores.append(0.0)
            if Np == 0:
                n_frames_no_pred += 1
            # Still track class counts even when one side is empty
            for c in gt[:, 0] if Ng > 0 else []:
                class_gt_count[int(c)] += 1
            for c in pred[:, 0] if Np > 0 else []:
                class_pred_count[int(c)] += 1
            continue

        pred_cls, pred_xyxy = yolo_to_xyxy_torch(pred, H, W)
        gt_cls, gt_xyxy = yolo_to_xyxy_torch(gt, H, W)

        for c in gt_cls.tolist():
            class_gt_count[int(c)] += 1
        for c in pred_cls.tolist():
            class_pred_count[int(c)] += 1

        frame_iou = 0.0

        for c in torch.unique(gt_cls):
            c_int = int(c)
            gt_mask = gt_cls == c
            pr_mask = pred_cls == c
            n_pr_c = int(pr_mask.sum())

            if n_pr_c == 0:
                # All GT boxes for this class get IoU=0, nothing to add
                continue

            ious = box_iou(gt_xyxy[gt_mask], pred_xyxy[pr_mask])  # [n_gt, n_pred]
            matched_ious = _greedy_match_ious(ious)
            matched_sum = float(sum(matched_ious))

            frame_iou += matched_sum
            class_iou_sum[c_int] += matched_sum
            class_tp[c_int] += sum(i > iou_threshold for i in matched_ious)

        frame_scores.append(frame_iou / max(Ng, Np))

    # Assemble per-class stats
    all_classes = set(class_gt_count) | set(class_pred_count)
    per_class = {}
    for c in sorted(all_classes):
        gt_c = class_gt_count[c]
        pr_c = class_pred_count[c]
        tp_c = class_tp[c]
        per_class[c] = ClassStats(
            mean_iou=class_iou_sum[c] / gt_c if gt_c > 0 else 0.0,
            recall=tp_c / gt_c if gt_c > 0 else 0.0,
            precision=tp_c / pr_c if pr_c > 0 else 0.0,
        )

    return EvalStats(
        score=float(np.mean(frame_scores)),
        per_class=per_class,
        n_frames=len(frame_scores),
        n_frames_both_empty=n_frames_both_empty,
        n_frames_no_pred=n_frames_no_pred,
    )

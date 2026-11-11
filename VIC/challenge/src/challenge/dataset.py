from collections.abc import Iterable
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

YOLO_DIM = 5
BBOXES_NUMPY_DIM = 2
PATCH_DIM = 3

class_names = {
    -1: "",
    0: "Hammer",
    1: "Knife",
    2: "Gun",
    3: "Wrench",
    4: "HandCuffs",
    5: "Bullet",
}


class Sample(TypedDict):
    image: np.ndarray
    targets: np.ndarray  # in Yolo format
    id: str


###### Motivation of this approach
# Loading the whole dataset in memory is quite ram expensive and slow
# Inspired by the torch dataset implementation we load only the yaml
# file and it's paths and only load the images on the fly when prompted
# Dataset is an iterable of the Sampled Typed dict containing
# 1. Images (always)
# 2. Targets (if they exist and asked)
# 3. Ids (only used to write the test submissions)
class Dataset:
    def __init__(
        self,
        meta_path: str,
        split: str = "train",
        with_labels: bool = True,
        with_ids: bool = True,
    ):
        with open(meta_path) as f:
            self.metadata = yaml.safe_load(f)

        self.image_paths = sorted(
            (Path(self.metadata["path"]) / self.metadata[split]).glob("*.jpg"),
        )
        self.with_labels = with_labels
        self.with_ids = with_ids

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Sample:
        image_path = self.image_paths[idx]

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

        boxes = []
        if self.with_labels:
            label_path = Path(
                str(image_path.with_suffix(".txt")).replace("images", "labels"),
            )
            with open(label_path) as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.split())
                    boxes.append((int(cls), cx, cy, w, h))

        targets = np.array(boxes, dtype=np.float32).reshape(-1, 5)

        return {"image": image, "targets": targets, "id": str(image_path.stem)}

    def __iter__(self):  # for the type checker
        for idx in range(len(self)):
            yield self[idx]


############ Submission utilities ############
def create_submission(
    loader: Iterable[Sample],
    pred_boxes: list,
    output_csv: str | Path,
):
    data = []
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    for sample, bboxes in zip(loader, pred_boxes):
        frame_id = sample["id"]

        if len(bboxes) == 0:
            bbs_str = "[]"
        else:
            bbs = bboxes.astype(float).tolist()
            bbs_str = str(bbs)

        data.append(
            {
                "frame_id": frame_id,
                "bbs": bbs_str,
            }
        )

    pd.DataFrame(data).to_csv(output_csv, index=False)


######## Utils for working with bboxs #########
def xywh_to_yolo(
    bboxs: list | np.ndarray,
    img_shape: tuple,
    cls_id: Optional[int | list] = None,
) -> np.ndarray:
    """Bboxs : (4,) or (N,4)
    cls_id : int (same for all), array (must be of same len as bbox dim 1, or nothing).
    """

    H, W = img_shape[:2]
    bboxs = np.asarray(bboxs, dtype=np.float32)

    N = bboxs.shape[0]
    x, y, w, h = bboxs.T

    cx = (x + 0.5 * w) / W
    cy = (y + 0.5 * h) / H
    bw = w / W
    bh = h / H

    if cls_id is None:
        cls = np.full(N, -1, dtype=np.float32)  # background is -1
    else:
        cls = np.asarray(cls_id, dtype=np.float32)

        if cls.ndim == 0:
            cls = np.full(N, cls, dtype=np.float32)

    return np.stack([cls, cx, cy, bw, bh], axis=1)


def xywh_to_xyxy(bboxs: np.ndarray) -> np.ndarray:
    # bboxs (N, 4)
    return np.stack(
        [
            bboxs[:, 0],
            bboxs[:, 1],
            bboxs[:, 0] + bboxs[:, 2],
            bboxs[:, 1] + bboxs[:, 3],
        ],
        axis=1,
    )


def yolo_to_xyxy(bboxs: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    # bboxs is (N,5)
    labels = bboxs[:, 0]
    cx = bboxs[:, 1] * W
    cy = bboxs[:, 2] * H
    bw = bboxs[:, 3] * W
    bh = bboxs[:, 4] * H

    gt_xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return labels, gt_xyxy


# This one is important because we use the torch vision boxiou function
# instead of writing our own, so we need to move everything to torch
def yolo_to_xyxy_torch(
    bboxs: np.ndarray | torch.Tensor, H: int, W: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(bboxs, torch.Tensor):
        if bboxs.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 4), dtype=torch.float32),
            )
        bboxs_t = bboxs
        labels_t = bboxs_t[:, 0].to(dtype=torch.long)
        cx = bboxs_t[:, 1] * W
        cy = bboxs_t[:, 2] * H
        bw = bboxs_t[:, 3] * W
        bh = bboxs_t[:, 4] * H
        gt_xyxy_t = torch.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
            dim=1,
        ).to(dtype=torch.float32)
        return labels_t, gt_xyxy_t

    labels, gt_xyxy = yolo_to_xyxy(bboxs, H, W)
    gt_xyxy_t = torch.as_tensor(gt_xyxy, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    return labels_t, gt_xyxy_t

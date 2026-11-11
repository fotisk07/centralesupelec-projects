import json
import pickle
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import ops
from tqdm import tqdm
from xgboost import XGBClassifier

from challenge.dataset import (
    Dataset,
    xywh_to_xyxy,
    xywh_to_yolo,
    yolo_to_xyxy_torch,
)
from challenge.metrics import EvalStats, _greedy_match_ious, evaluate_model_with_stats

###################################### DISCLAIMER ######################################
# THe print statements as well as the loading/caching is AI generated
# We use torch not for deep learning but because the torchvision library has a vectorised
# box_iou implementation
##########################################################################################


@dataclass
class Config:
    data_yaml: str = "data/data.yaml"

    # Proposals
    canny1: int = 80
    canny2: int = 160
    morph_kernel_size: int = 7
    min_area_frac: float = 0.001
    max_area_frac: float = 0.45
    train_proposal_cap: int = 140
    infer_proposal_cap: int = 56
    nms_iou_threshold: float = 0.45
    use_edge_proposals: bool = True
    use_adaptive_proposals: bool = True
    use_mser_proposals: bool = True

    # Labels and sampling
    positive_iou_threshold: float = 0.30
    negative_iou_threshold: float = 0.10
    background_class_id: int = -1
    max_background_to_positive_ratio: float = 2.5

    # Features
    patch_size: int = 48
    color_bins: int = 8
    grad_bins: int = 8
    hog_grid_size: int = 4
    hog_bins: int = 9

    # Classifier
    class_model_type: str = "xgboost"  # "xgboost" | "logreg"
    lr_max_iter: int = 3000
    lr_class_weight: str = "balanced"
    xgb_n_estimators: int = 350
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9

    # Inference thresholds
    max_boxes: int = 1
    submission_max_boxes: int = 1
    objectness_threshold: float = 0.20
    class_threshold: float = 0.05
    final_score_threshold: float = 0.05
    per_class_class_thresholds: tuple[float, ...] = ()
    per_class_final_score_thresholds: tuple[float, ...] = ()

    # Runtime
    train_feature_fraction: float = 1.0
    train_features_cache_path: str = "outputs/cache/train_features_simple.pkl"
    load_train_features_cache: bool = True
    save_train_features_cache: bool = True
    eval_iou_threshold: float = 0.30

    # Config loading/output
    best_cfg_path: str = "outputs/cache/best_cfg.json"
    load_best_cfg: bool = True
    generate_submission: bool = True
    submission_out: str = "outputs/submission_simple.csv"


class ProposalFeatureExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._mser = cv2.MSER_create()

    def _valid_bbox(self, bbox: tuple[int, int, int, int], H: int, W: int) -> bool:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return False

        area = w * h
        return self.cfg.min_area_frac * H * W <= area <= self.cfg.max_area_frac * H * W

    def _edge_proposals(
        self, img: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.cfg.canny1, self.cfg.canny2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size),
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        for c in cnts:
            bbox = cv2.boundingRect(c)
            if self._valid_bbox(bbox, H, W):
                out.append((bbox, c))
        return out

    def _adaptive_proposals(
        self, img: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thr = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        for c in cnts:
            bbox = cv2.boundingRect(c)
            if self._valid_bbox(bbox, H, W):
                out.append((bbox, c))
        return out

    def _mser_proposals(
        self, img: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        regions, _ = self._mser.detectRegions(gray)

        out: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        for pts in regions:
            if len(pts) < 10:
                continue
            contour = pts.reshape(-1, 1, 2).astype(np.int32)
            bbox = cv2.boundingRect(contour)
            if self._valid_bbox(bbox, H, W):
                out.append((bbox, contour))
        return out

    def propose(
        self,
        img: np.ndarray,
        *,
        proposal_cap: int,
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        proposals: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

        # Merge complementary proposal sources to improve recall.
        if self.cfg.use_edge_proposals:
            proposals.extend(self._edge_proposals(img))
        if self.cfg.use_adaptive_proposals:
            proposals.extend(self._adaptive_proposals(img))
        if self.cfg.use_mser_proposals:
            proposals.extend(self._mser_proposals(img))

        if not proposals:
            return []

        boxes_xyxy = []
        scores = []
        for bbox, _ in proposals:
            x, y, w, h = bbox
            boxes_xyxy.append([x, y, x + w, y + h])
            scores.append(float(w * h))

        # Deduplicate overlapping candidates before downstream scoring.
        keep = ops.nms(
            torch.as_tensor(boxes_xyxy, dtype=torch.float32),
            torch.as_tensor(scores, dtype=torch.float32),
            self.cfg.nms_iou_threshold,
        )
        keep = keep[:proposal_cap].tolist()
        return [proposals[i] for i in keep]

    def _compact_hog_features(self, mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
        grid = max(1, self.cfg.hog_grid_size)
        bins = max(1, self.cfg.hog_bins)
        H, W = mag.shape
        cell_h = max(1, H // grid)
        cell_w = max(1, W // grid)
        parts: list[np.ndarray] = []

        for gy in range(grid):
            for gx in range(grid):
                y0 = gy * cell_h
                x0 = gx * cell_w
                y1 = H if gy == grid - 1 else (gy + 1) * cell_h
                x1 = W if gx == grid - 1 else (gx + 1) * cell_w

                cell_mag = mag[y0:y1, x0:x1].reshape(-1)
                cell_ang = ang[y0:y1, x0:x1].reshape(-1)
                if len(cell_mag) == 0:
                    hist = np.zeros(bins, dtype=np.float32)
                else:
                    hist, _ = np.histogram(
                        cell_ang,
                        bins=bins,
                        range=(0.0, 180.0),
                        weights=cell_mag,
                    )
                    hist = hist.astype(np.float32)
                    hist /= hist.sum() + 1e-6
                parts.append(hist)

        return np.concatenate(parts, axis=0)

    def _patch_features(
        self,
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> dict[str, float]:
        x, y, w, h = bbox
        patch = img[y : y + h, x : x + w]
        if patch.size == 0:
            patch = np.zeros(
                (self.cfg.patch_size, self.cfg.patch_size, 3), dtype=np.uint8
            )

        patch = cv2.resize(
            patch,
            (self.cfg.patch_size, self.cfg.patch_size),
            interpolation=cv2.INTER_LINEAR,
        )

        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = cv2.phase(gx, gy, angleInDegrees=True) % 180.0

        grad_hist, _ = np.histogram(
            ang.flatten(),
            bins=self.cfg.grad_bins,
            range=(0.0, 180.0),
            weights=mag.flatten() + 1e-6,
        )
        grad_hist = grad_hist / (grad_hist.sum() + 1e-6)

        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist(
            [hsv], [0], None, [self.cfg.color_bins], [0, 180]
        ).flatten()
        s_hist = cv2.calcHist(
            [hsv], [1], None, [self.cfg.color_bins], [0, 256]
        ).flatten()
        v_hist = cv2.calcHist(
            [hsv], [2], None, [self.cfg.color_bins], [0, 256]
        ).flatten()
        h_hist /= h_hist.sum() + 1e-6
        s_hist /= s_hist.sum() + 1e-6
        v_hist /= v_hist.sum() + 1e-6

        lbp = np.zeros_like(gray, dtype=np.uint8)
        center = gray[1:-1, 1:-1]
        neighbors = [
            gray[:-2, :-2],
            gray[:-2, 1:-1],
            gray[:-2, 2:],
            gray[1:-1, 2:],
            gray[2:, 2:],
            gray[2:, 1:-1],
            gray[2:, :-2],
            gray[1:-1, :-2],
        ]
        for k, n in enumerate(neighbors):
            lbp[1:-1, 1:-1] |= ((n >= center) << k).astype(np.uint8)
        lbp_hist = np.bincount(lbp.flatten(), minlength=16)[:16].astype(np.float32)
        lbp_hist /= lbp_hist.sum() + 1e-6

        hog_hist = self._compact_hog_features(mag, ang)

        feats: dict[str, float] = {
            "patch_mean": float(gray.mean()),
            "patch_std": float(gray.std()),
            "edge_density": float((mag > mag.mean()).mean()),
        }
        for i, v in enumerate(grad_hist):
            feats[f"grad_{i}"] = float(v)
        for i, v in enumerate(h_hist):
            feats[f"h_hist_{i}"] = float(v)
        for i, v in enumerate(s_hist):
            feats[f"s_hist_{i}"] = float(v)
        for i, v in enumerate(v_hist):
            feats[f"v_hist_{i}"] = float(v)
        for i, v in enumerate(lbp_hist):
            feats[f"lbp_{i}"] = float(v)
        for i, v in enumerate(hog_hist):
            feats[f"hog_{i}"] = float(v)

        return feats

    def extract_features(
        self,
        img: np.ndarray,
        contour: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> dict[str, float]:
        H, W = img.shape[:2]
        x, y, w, h = bbox
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        hull_area = cv2.contourArea(cv2.convexHull(contour))

        feats: dict[str, float] = {
            "bbox_aspect_ratio": max(w, h) / (min(w, h) + 1e-6),
            "bbox_area": float(w * h),
            "bbox_area_norm": float((w * h) / (H * W + 1e-6)),
            "bbox_w_norm": float(w / (W + 1e-6)),
            "bbox_h_norm": float(h / (H + 1e-6)),
            "bbox_center_x_norm": float((x + 0.5 * w) / (W + 1e-6)),
            "bbox_center_y_norm": float((y + 0.5 * h) / (H + 1e-6)),
            "contour_area": float(area),
            "extent": float(area / (w * h + 1e-6)),
            "solidity": float(area / (hull_area + 1e-6)),
            "circularity": float(4 * np.pi * area / (perimeter**2 + 1e-6)),
        }

        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten() if moments["m00"] > 0 else np.zeros(7)
        for i, v in enumerate(hu):
            feats[f"hu_{i}"] = float(-np.sign(v) * np.log10(abs(v) + 1e-10))

        feats.update(self._patch_features(img, bbox))
        return feats

    def _gt_to_xyxy(self, gt: np.ndarray, H: int, W: int) -> torch.Tensor:
        gt_xywh_px = np.stack(
            [
                gt[:, 1] * W - 0.5 * gt[:, 3] * W,
                gt[:, 2] * H - 0.5 * gt[:, 4] * H,
                gt[:, 3] * W,
                gt[:, 4] * H,
            ],
            axis=1,
        )
        return torch.as_tensor(xywh_to_xyxy(gt_xywh_px), dtype=torch.float32)

    def build_training_samples(
        self,
        loader: Dataset,
        *,
        fraction: float,
        positive_iou_threshold: float,
        negative_iou_threshold: float,
    ) -> tuple[list[dict[str, float]], list[int], list[int]]:
        X_feats: list[dict[str, float]] = []
        y_obj: list[int] = []
        y_cls: list[int] = []

        n = max(1, int(len(loader) * float(np.clip(fraction, 0.0, 1.0))))
        for idx in tqdm(range(n), total=n):
            sample = loader[idx]
            img = sample["image"]
            gt = sample["targets"]
            if len(gt) == 0:
                continue

            H, W = img.shape[:2]
            gt_xyxy = self._gt_to_xyxy(gt, H, W)
            proposals = self.propose(img, proposal_cap=self.cfg.train_proposal_cap)

            for bbox, contour in proposals:
                prop_xyxy = torch.as_tensor(
                    xywh_to_xyxy(np.array([bbox], dtype=np.float32)),
                    dtype=torch.float32,
                )
                ious = ops.box_iou(prop_xyxy, gt_xyxy)[0]
                best_idx = int(ious.argmax().item())
                best_iou = float(ious[best_idx].item())

                # Label by best-IoU match: positive / background / ignore band.
                if best_iou >= positive_iou_threshold:
                    obj_label = 1
                    cls_label = int(gt[best_idx, 0])
                elif best_iou <= negative_iou_threshold:
                    obj_label = 0
                    cls_label = self.cfg.background_class_id
                else:
                    continue

                X_feats.append(self.extract_features(img, contour, bbox))
                y_obj.append(obj_label)
                y_cls.append(cls_label)

        return X_feats, y_obj, y_cls


class ProposalClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.obj_vec = DictVectorizer(sparse=False)
        self.cls_vec = DictVectorizer(sparse=False)
        self.obj_clf = None
        self.cls_clf = None
        self.is_fitted = False

    def _build_logreg(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=self.cfg.lr_max_iter,
                        class_weight=self.cfg.lr_class_weight,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def _build_cls_model(self, n_classes: int):
        if self.cfg.class_model_type == "xgboost":
            return XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                n_estimators=self.cfg.xgb_n_estimators,
                max_depth=self.cfg.xgb_max_depth,
                learning_rate=self.cfg.xgb_learning_rate,
                subsample=self.cfg.xgb_subsample,
                colsample_bytree=self.cfg.xgb_colsample_bytree,
                eval_metric="mlogloss",
                n_jobs=8,
                random_state=42,
            )

        return self._build_logreg()

    def _downsample_background(
        self,
        X_feats: list[dict[str, float]],
        y_obj: list[int],
        y_cls: list[int],
    ) -> tuple[list[dict[str, float]], list[int], list[int]]:
        pos_idx = [i for i, y in enumerate(y_obj) if y == 1]
        bg_idx = [i for i, y in enumerate(y_obj) if y == 0]
        if not pos_idx or not bg_idx:
            return X_feats, y_obj, y_cls

        max_bg = int(len(pos_idx) * self.cfg.max_background_to_positive_ratio)
        if len(bg_idx) <= max_bg:
            return X_feats, y_obj, y_cls

        rng = np.random.default_rng(42)
        keep_bg = rng.choice(bg_idx, size=max_bg, replace=False).tolist()
        keep_idx = sorted(pos_idx + keep_bg)
        return (
            [X_feats[i] for i in keep_idx],
            [y_obj[i] for i in keep_idx],
            [y_cls[i] for i in keep_idx],
        )

    def fit(
        self,
        X_feats: list[dict[str, float]],
        y_obj: list[int],
        y_cls: list[int],
    ) -> None:
        print(
            f"[fit/classifier] Preparing samples: {len(X_feats)} "
            f"(positives={sum(y_obj)}, background={len(y_obj) - sum(y_obj)})"
        )
        X_feats, y_obj, y_cls = self._downsample_background(X_feats, y_obj, y_cls)
        print(
            f"[fit/classifier] After downsampling: {len(X_feats)} "
            f"(positives={sum(y_obj)}, background={len(y_obj) - sum(y_obj)})"
        )
        if not X_feats:
            raise RuntimeError("No training features to fit classifier.")

        # Stage 1: binary objectness model.
        print("[fit/classifier] Training objectness model...")
        X_obj = self.obj_vec.fit_transform(X_feats)
        self.obj_clf = self._build_logreg()
        self.obj_clf.fit(X_obj, y_obj)

        # Stage 2: class model trained only on positives.
        pos_idx = [i for i, y in enumerate(y_obj) if y == 1]
        X_cls_feats = [X_feats[i] for i in pos_idx]
        y_cls_pos = [y_cls[i] for i in pos_idx]
        X_cls = self.cls_vec.fit_transform(X_cls_feats)

        n_cls = len(set(y_cls_pos))
        print(
            f"[fit/classifier] Training class model on {len(X_cls_feats)} positives "
            f"across {n_cls} classes..."
        )
        self.cls_clf = self._build_cls_model(max(2, n_cls))
        self.cls_clf.fit(X_cls, y_cls_pos)

        self.is_fitted = True
        print("[fit/classifier] Training complete.")

    def predict_scores(
        self,
        X_feats: list[dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction.")

        X_obj = self.obj_vec.transform(X_feats)
        obj_probs = self.obj_clf.predict_proba(X_obj)
        obj_classes = np.asarray(self.obj_clf.classes_)
        obj_pos_idx = int(np.where(obj_classes == 1)[0][0])
        obj_conf = obj_probs[:, obj_pos_idx].astype(np.float32)

        X_cls = self.cls_vec.transform(X_feats)
        cls_probs = self.cls_clf.predict_proba(X_cls)
        cls_classes = np.asarray(self.cls_clf.classes_)
        cls_best_idx = cls_probs.argmax(axis=1)
        cls_conf = cls_probs[np.arange(len(cls_probs)), cls_best_idx].astype(np.float32)
        best_cls = cls_classes[cls_best_idx].astype(np.int32)

        # Final proposal confidence used for ranking and thresholding.
        final_score = (obj_conf * cls_conf).astype(np.float32)
        return obj_conf, cls_conf, best_cls, final_score


class DetectorModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.extractor = ProposalFeatureExtractor(cfg)
        self.classifier = ProposalClassifier(cfg)
        self.is_fitted = False

    def _save_training_cache(
        self,
        X_feats: list[dict[str, float]],
        y_obj: list[int],
        y_cls: list[int],
    ) -> None:
        path = Path(self.cfg.train_features_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"X_feats": X_feats, "y_obj": y_obj, "y_cls": y_cls}, f)

    def _load_training_cache(
        self,
    ) -> tuple[list[dict[str, float]], list[int], list[int]] | None:
        path = Path(self.cfg.train_features_cache_path)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa:S301
        return payload["X_feats"], payload["y_obj"], payload["y_cls"]

    def _class_threshold_arrays(
        self,
        class_ids: np.ndarray,
        *,
        class_threshold: float,
        final_score_threshold: float,
        per_class_class_thresholds: tuple[float, ...],
        per_class_final_score_thresholds: tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        cls_thr = np.full(len(class_ids), class_threshold, dtype=np.float32)
        final_thr = np.full(len(class_ids), final_score_threshold, dtype=np.float32)

        if len(per_class_class_thresholds) == 6:
            for cid, thr in enumerate(per_class_class_thresholds):
                cls_thr[class_ids == cid] = float(thr)
        if len(per_class_final_score_thresholds) == 6:
            for cid, thr in enumerate(per_class_final_score_thresholds):
                final_thr[class_ids == cid] = float(thr)

        return cls_thr, final_thr

    def fit(self, train_loader: Dataset) -> None:
        print("[fit] Starting detector training...")
        loaded = None
        if self.cfg.load_train_features_cache:
            print(
                f"[fit] Loading feature cache from {self.cfg.train_features_cache_path}..."
            )
            loaded = self._load_training_cache()

        if loaded is None:
            if self.cfg.load_train_features_cache:
                print(
                    "[fit] Cache not found, building training features from dataset..."
                )
            else:
                print(
                    "[fit] Cache loading disabled, building training features from dataset..."
                )
            X_feats, y_obj, y_cls = self.extractor.build_training_samples(
                train_loader,
                fraction=self.cfg.train_feature_fraction,
                positive_iou_threshold=self.cfg.positive_iou_threshold,
                negative_iou_threshold=self.cfg.negative_iou_threshold,
            )
            print(
                f"[fit] Built {len(X_feats)} feature samples "
                f"(positives={sum(y_obj)}, background={len(y_obj) - sum(y_obj)})."
            )
            if self.cfg.save_train_features_cache:
                print(
                    f"[fit] Saving feature cache to {self.cfg.train_features_cache_path}..."
                )
                self._save_training_cache(X_feats, y_obj, y_cls)
                print("[fit] Feature cache saved.")
        else:
            # Reuse cached training features to skip expensive proposal extraction.
            X_feats, y_obj, y_cls = loaded
            print(
                f"[fit] Loaded {len(X_feats)} cached feature samples "
                f"(positives={sum(y_obj)}, background={len(y_obj) - sum(y_obj)})."
            )

        print("[fit] Fitting classifiers...")
        self.classifier.fit(X_feats, y_obj, y_cls)
        self.is_fitted = True
        print("[fit] Detector training complete.")

    def analyze_sample(
        self,
        img: np.ndarray,
        *,
        max_boxes: int | None = None,
    ) -> dict[str, np.ndarray | list]:
        cap = max_boxes or self.cfg.max_boxes
        proposal_cap = max(self.cfg.infer_proposal_cap, cap * 6)
        proposals = self.extractor.propose(img, proposal_cap=proposal_cap)

        if not proposals:
            return {
                "boxes": [],
                "best_class": np.array([], dtype=np.int32),
                "obj_conf": np.array([], dtype=np.float32),
                "cls_conf": np.array([], dtype=np.float32),
                "final_score": np.array([], dtype=np.float32),
                "keep_idx": np.array([], dtype=np.int64),
            }

        boxes: list[tuple[int, int, int, int]] = []
        feats: list[dict[str, float]] = []
        for box, contour in proposals:
            boxes.append(box)
            feats.append(self.extractor.extract_features(img, contour, box))

        obj_conf, cls_conf, best_cls, final_score = self.classifier.predict_scores(
            feats
        )
        cls_thr, final_thr = self._class_threshold_arrays(
            best_cls,
            class_threshold=self.cfg.class_threshold,
            final_score_threshold=self.cfg.final_score_threshold,
            per_class_class_thresholds=self.cfg.per_class_class_thresholds,
            per_class_final_score_thresholds=self.cfg.per_class_final_score_thresholds,
        )

        # Apply thresholds, then keep top-k proposals by final score.
        keep_mask = (
            (obj_conf >= self.cfg.objectness_threshold)
            & (cls_conf >= cls_thr)
            & (final_score >= final_thr)
        )
        keep_idx = np.where(keep_mask)[0]
        if len(keep_idx) > 0:
            keep_idx = keep_idx[np.argsort(-final_score[keep_idx])]
            keep_idx = keep_idx[:cap]

        return {
            "boxes": boxes,
            "best_class": best_cls,
            "obj_conf": obj_conf,
            "cls_conf": cls_conf,
            "final_score": final_score,
            "keep_idx": keep_idx,
        }

    def _predict_single(self, img: np.ndarray, *, max_boxes: int) -> np.ndarray:
        out = self.analyze_sample(img, max_boxes=max_boxes)
        keep_idx = out["keep_idx"]
        if len(keep_idx) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        boxes = out["boxes"]
        classes = out["best_class"][keep_idx].tolist()
        final_boxes = [boxes[i] for i in keep_idx]
        return xywh_to_yolo(final_boxes, img.shape, cls_id=classes)

    def predict(
        self, loader: Dataset, *, max_boxes: int | None = None
    ) -> list[np.ndarray]:
        cap = max_boxes or self.cfg.max_boxes
        preds: list[np.ndarray] = []
        for sample in tqdm(loader):
            preds.append(self._predict_single(sample["image"], max_boxes=cap))  # noqa : PERF401
        return preds

    def evaluate(self, pred_boxes: list[np.ndarray], loader: Dataset) -> EvalStats:
        return evaluate_model_with_stats(
            pred_boxes,
            loader,
            iou_threshold=self.cfg.eval_iou_threshold,
        )

    def prepare_analysis_cache(
        self,
        loader: Dataset,
        *,
        fraction: float = 1.0,
        max_boxes: int | None = None,
    ) -> list[dict[str, object]]:
        # Cache proposal-level outputs once; threshold sweeps can then run cheaply.
        n = max(1, int(len(loader) * float(np.clip(fraction, 0.0, 1.0))))
        cache: list[dict[str, object]] = []
        for idx in tqdm(range(n), total=n):
            sample = loader[idx]
            img = sample["image"]
            H, W = img.shape[:2]
            cache.append(
                {
                    "analysis": self.analyze_sample(img, max_boxes=max_boxes),
                    "targets": sample["targets"],
                    "H": H,
                    "W": W,
                }
            )
        return cache

    def predictions_from_analysis_cache(
        self,
        cache: list[dict[str, object]],
        *,
        objectness_threshold: float,
        class_threshold: float,
        final_score_threshold: float,
        max_boxes: int,
        per_class_class_thresholds: tuple[float, ...] = (),
        per_class_final_score_thresholds: tuple[float, ...] = (),
    ) -> list[np.ndarray]:
        preds: list[np.ndarray] = []

        for row in cache:
            analysis = row["analysis"]
            boxes = analysis["boxes"]
            best_cls = analysis["best_class"]
            obj_conf = analysis["obj_conf"]
            cls_conf = analysis["cls_conf"]
            final_score = analysis["final_score"]
            H, W = int(row["H"]), int(row["W"])

            if len(boxes) == 0:
                preds.append(np.zeros((0, 5), dtype=np.float32))
                continue

            cls_thr, final_thr = self._class_threshold_arrays(
                best_cls,
                class_threshold=class_threshold,
                final_score_threshold=final_score_threshold,
                per_class_class_thresholds=per_class_class_thresholds,
                per_class_final_score_thresholds=per_class_final_score_thresholds,
            )
            keep_mask = (
                (obj_conf >= objectness_threshold)
                & (cls_conf >= cls_thr)
                & (final_score >= final_thr)
            )
            keep_idx = np.where(keep_mask)[0]
            if len(keep_idx) > 0:
                keep_idx = keep_idx[np.argsort(-final_score[keep_idx])]
                keep_idx = keep_idx[:max_boxes]

            if len(keep_idx) == 0:
                preds.append(np.zeros((0, 5), dtype=np.float32))
                continue

            final_boxes = [boxes[i] for i in keep_idx]
            final_classes = best_cls[keep_idx].tolist()
            preds.append(xywh_to_yolo(final_boxes, (H, W, 3), cls_id=final_classes))

        return preds

    def score_from_analysis_cache(
        self,
        cache: list[dict[str, object]],
        preds: list[np.ndarray],
    ) -> float:
        frame_scores: list[float] = []

        for row, pred in zip(cache, preds):
            gt = row["targets"]
            H, W = int(row["H"]), int(row["W"])
            Ng, Np = len(gt), len(pred)

            if Ng == 0 and Np == 0:
                frame_scores.append(1.0)
                continue
            if Ng == 0 or Np == 0:
                frame_scores.append(0.0)
                continue

            pred_cls, pred_xyxy = yolo_to_xyxy_torch(pred, H, W)
            gt_cls, gt_xyxy = yolo_to_xyxy_torch(gt, H, W)
            frame_iou = 0.0

            # Per-class greedy one-to-one IoU matching.
            for c in torch.unique(gt_cls):
                gt_mask = gt_cls == c
                pr_mask = pred_cls == c
                if int(pr_mask.sum()) == 0:
                    continue

                ious = ops.box_iou(gt_xyxy[gt_mask], pred_xyxy[pr_mask])
                matched = _greedy_match_ious(ious)
                frame_iou += float(sum(matched))

            frame_scores.append(frame_iou / max(Ng, Np))

        return float(np.mean(frame_scores)) if frame_scores else 0.0


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def load_config_overrides(cfg: Config, cfg_path: str | Path) -> Config:
    path = Path(cfg_path)

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    valid_fields = {f.name for f in fields(Config)}
    tuple_float_fields = {
        "per_class_class_thresholds",
        "per_class_final_score_thresholds",
    }

    # Loading JSON to dataclass can sometimes create problems...
    for key, value in payload.items():
        if key not in valid_fields:
            continue

        current = getattr(cfg, key)
        if key in tuple_float_fields:
            setattr(cfg, key, tuple(float(v) for v in value))
        elif isinstance(current, bool):
            setattr(cfg, key, _coerce_bool(value))
        elif isinstance(current, int):
            setattr(cfg, key, int(value))
        elif isinstance(current, float):
            setattr(cfg, key, float(value))
        else:
            setattr(cfg, key, value)

    return cfg


def load_best_config(cfg: Config) -> Config:
    best_path = Path(cfg.best_cfg_path)
    if best_path.exists():
        return load_config_overrides(cfg, best_path)
    return cfg


def resolve_config(
    cfg_path: str | Path = "",
    *,
    base_cfg: Config | None = None,
    submission_out: str = "",
) -> Config:
    cfg = base_cfg or Config()
    if cfg_path:
        cfg = load_config_overrides(cfg, cfg_path)
    elif cfg.load_best_cfg:
        cfg = load_best_config(cfg)

    if submission_out:
        cfg.submission_out = submission_out
        cfg.generate_submission = True

    return cfg

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from tqdm import tqdm

from challenge.dataset import Dataset
from challenge.models import Config, DetectorModel, resolve_config


@dataclass
class TuneConfig:
    objectness_thresholds: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35)
    class_thresholds: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.15)
    final_score_thresholds: tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.15)
    max_boxes_values: tuple[int, ...] = (1, 2, 3)
    val_fraction: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def score_with_thresholds(
    model: DetectorModel,
    cache: list[dict[str, object]],
    *,
    objectness_threshold: float,
    class_threshold: float,
    final_score_threshold: float,
    max_boxes: int,
) -> float:
    preds = model.predictions_from_analysis_cache(
        cache,
        objectness_threshold=objectness_threshold,
        class_threshold=class_threshold,
        final_score_threshold=final_score_threshold,
        max_boxes=max_boxes,
    )
    return model.score_from_analysis_cache(cache, preds)


def build_payload(
    cfg: Config,
    *,
    objectness_threshold: float,
    class_threshold: float,
    final_score_threshold: float,
    max_boxes: int,
    tune_score: float,
) -> dict[str, object]:
    payload = asdict(cfg)
    payload.update(
        {
            "objectness_threshold": float(objectness_threshold),
            "class_threshold": float(class_threshold),
            "final_score_threshold": float(final_score_threshold),
            "max_boxes": int(max_boxes),
            "best_tune_score": float(tune_score),
        }
    )
    return payload


def main() -> None:
    args = parse_args()

    cfg = resolve_config(
        args.cfg_path,
        base_cfg=Config(
            generate_submission=False,
            load_best_cfg=False,
            load_train_features_cache=True,
            save_train_features_cache=True,
        ),
    )

    train_data = Dataset(cfg.data_yaml, split="train")
    val_data = Dataset(cfg.data_yaml, split="val")

    model = DetectorModel(cfg)
    model.fit(train_data)

    tune_cfg = TuneConfig()
    print("Preparing validation cache...")
    cache = model.prepare_analysis_cache(
        val_data,
        fraction=tune_cfg.val_fraction,
        max_boxes=max(tune_cfg.max_boxes_values),
    )

    combos = list(
        product(
            tune_cfg.objectness_thresholds,
            tune_cfg.class_thresholds,
            tune_cfg.final_score_thresholds,
            tune_cfg.max_boxes_values,
        )
    )
    print(f"Running {len(combos)} threshold combinations...")

    best_score = -1.0
    best_combo = None

    for obj_thr, cls_thr, final_thr, max_boxes in tqdm(combos, total=len(combos)):
        score = score_with_thresholds(
            model,
            cache,
            objectness_threshold=obj_thr,
            class_threshold=cls_thr,
            final_score_threshold=final_thr,
            max_boxes=max_boxes,
        )
        if score > best_score:
            best_score = float(score)
            best_combo = (obj_thr, cls_thr, final_thr, max_boxes)

    best_obj, best_cls, best_final, best_max_boxes = best_combo
    payload = build_payload(
        cfg,
        objectness_threshold=best_obj,
        class_threshold=best_cls,
        final_score_threshold=best_final,
        max_boxes=best_max_boxes,
        tune_score=best_score,
    )

    out_path = Path(args.out) if args.out else Path(cfg.best_cfg_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Best tuning result:")
    print(f"  score={best_score:.4f}")
    print(f"  objectness_threshold={best_obj}")
    print(f"  class_threshold={best_cls}")
    print(f"  final_score_threshold={best_final}")
    print(f"  max_boxes={best_max_boxes}")
    print(f"Saved best config -> {out_path}")


if __name__ == "__main__":
    main()

import argparse

from challenge.dataset import Dataset, create_submission
from challenge.models import DetectorModel, resolve_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--out",
        "--submission-out",
        dest="submission_out",
        type=str,
        default="",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.cfg_path, submission_out=args.submission_out)

    train_data = Dataset(cfg.data_yaml, split="train")
    val_data = Dataset(cfg.data_yaml, split="val")

    model = DetectorModel(cfg)
    model.fit(train_data)

    print("\nGenerating val predictions...")
    val_preds = model.predict(val_data, max_boxes=cfg.max_boxes)

    print("\nEvaluating val predictions...")
    stats = model.evaluate(val_preds, val_data)
    print(stats)

    if cfg.generate_submission:
        print("Generating test predictions...")
        test_data = Dataset(cfg.data_yaml, split="test", with_labels=False)
        test_preds = model.predict(test_data, max_boxes=cfg.submission_max_boxes)
        create_submission(test_data, test_preds, output_csv=cfg.submission_out)
        print(f"Submission saved to {cfg.submission_out}")


if __name__ == "__main__":
    main()

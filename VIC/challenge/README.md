# Challenge: Object Detection in X-Ray Baggage Images

Classical CV + ML pipeline for 6-class object detection in X-ray baggage images, including training, validation, threshold tuning, and submission export.

## Setup

Preferred (recommended): Use [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
```

Alternative:

```bash
pip install -r requirements.txt
```

## Data

Dataset metadata is read from `data/data.yaml`.  
Expected structure:

```yaml
path: data/raw
train: images/train
val: images/val
test: images/test
```

## Run

General command:

```bash
uv run python main.py
```

If you do not pass `--cfg-path`, the code tries to load the default best config by name: `outputs/cache/best_cfg.json`.

Optional explicit config:

```bash
uv run python main.py --cfg-path path/to/config.json
```

## Tuning Workflow

Typical workflow is:
1. Tune thresholds on validation.
2. Run the main pipeline with the best config.

Tuning command:

```bash
uv run python tune_validation.py --out outputs/cache/my_best_cfg.json
```

Because this repository already includes a config in `outputs/cache/best_cfg.json`, you can also skip tuning and run directly with `uv run python main.py`.


## Project Layout

- `main.py`: train, validate, and optionally generate test submission.
- `tune_validation.py`: validation threshold grid search.
- `src/challenge/models.py`: proposals, features, classifier, inference.
- `src/challenge/dataset.py`: dataset loading and submission formatting.
- `src/challenge/metrics.py`: evaluation metric and per-class stats.
- `data/data.yaml`: dataset paths and class names.
- `outputs/`: predictions, submissions, caches, and tuned configs.

## Disclaimer

Some `print` statements and parts of code refactoring were done with assistance from LLM tools.

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml.data_io import iter_signals_and_targets, iter_signals_only
from ml.features import add_epoch_drift, build_feature_df
from ml.model import cross_validate_xgboost, fit_decoder_context, predict_with_decoder, train_xgboost


@dataclass
class Config:  # noqa: D101
    bundle_dir: Path = Path("data")
    sf: int = 100  # sampling rate
    samples_per_epoch: int = 30 * 100  # 30s epochs
    stage_names: list[str] = field(default_factory=lambda: ["Wake", "N1", "N2", "N3", "REM"])
    # Feature params
    load_precomputed_features: bool = False
    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "slow_delta": (0.5, 2),
            "delta": (2, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "sigma": (12, 15),
            "beta": (15, 30),
        },
    )
    add_epoch_drift_window: int = 2
    add_epoch_drift_symmetric: bool = True
    # model params
    num_classes: int = 5
    eta: float = 0.03
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    num_boost_round: int = 500
    seed: int = 42
    smoothing_window: int = 1
    n_jobs: int = 4
    gamma: float = 1.5  # try 1.5 then 2.0
    # decoding
    decoder: str = "hmm"  # options: "argmax", "smooth", "hmm"
    hmm_alpha: float = 1.0
    # cross-validation params
    cv_n_splits: int = 5
    # choose whether to create submission predictions
    create_submission: bool = False
    submission_name: str = "submission.csv"


config = Config(
    load_precomputed_features=False,
    create_submission=True,
    submission_name="final.csv",
)

# -----------------------
# Load data and build feature DataFrames
# -----------------------
print("Loading raw data...")
raw_train_data = iter_signals_and_targets(config.bundle_dir, split="train")
raw_test_data = iter_signals_only(config.bundle_dir, split="test")
if config.load_precomputed_features:
    print("Loading precomputed feature DataFrames...")
    df_train = pd.read_csv("features/train_features.csv")
else:
    print("Building feature DataFrame for training data...")
    df_train = build_feature_df(raw_train_data, with_labels=True, config=config)
    df_train = add_epoch_drift(
        df_train,
        window=config.add_epoch_drift_window,
        symmetric=config.add_epoch_drift_symmetric,
    )

X = df_train.drop(columns=["label", "identifier", "patient_id", "epoch_idx"])
y = df_train["label"]
groups = df_train["patient_id"].to_numpy()


# -----------------------
# Train and evaluate model
# -----------------------
print("Starting cross-validation...")
cv_scores = cross_validate_xgboost(X, y, df_meta=df_train, groups=groups, config=config)

# -----------------------
# Create submission predictions
# -----------------------
if config.create_submission:
    print("Loading test features...")
    if config.load_precomputed_features:
        df_test = pd.read_csv("features/test_features.csv")
    else:
        print("Building feature DataFrame for test data...")
        df_test = build_feature_df(raw_test_data, with_labels=False, config=config)
        df_test = add_epoch_drift(
            df_test,
            window=config.add_epoch_drift_window,
            symmetric=config.add_epoch_drift_symmetric,
        )
    X_test = df_test.drop(columns=["identifier", "patient_id", "epoch_idx"])

    print("\nTraining final model on all training data...")
    model = train_xgboost(X, y, config)
    final_ctx = fit_decoder_context(
        y,
        df_train,
        config,
    )
    test_preds = predict_with_decoder(model, X_test, df_meta=df_test, config=config, ctx=final_ctx)
    submission_df = pd.DataFrame({"identifier": df_test["identifier"], "label": test_preds})
    submission_df.to_csv(config.submission_name, index=False)
    print("Done.")

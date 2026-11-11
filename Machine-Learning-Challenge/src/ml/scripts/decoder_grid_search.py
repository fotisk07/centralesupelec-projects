from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ml.model import cross_validate_xgboost


@dataclass
class Config:
    bundle_dir: Path = Path("data")

    # model (FIXED)
    num_classes: int = 5
    eta: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    num_boost_round: int = 500
    seed: int = 42
    n_jobs: int = 4

    # decoding (TO TUNE)
    decoder: str = "argmax"  # argmax | smooth | hmm
    smoothing_window: int = 3
    hmm_alpha: float = 1.0

    # CV
    cv_n_splits: int = 5

    stage_names: list[str] = field(default_factory=lambda: ["Wake", "N1", "N2", "N3", "REM"])


# ============================================================
# Load features once
# ============================================================

print("Loading features...")
df_train = pd.read_csv("features/train_features.csv")

X = df_train.drop(columns=["label", "identifier", "patient_id", "epoch_idx"])
y = df_train["label"]
groups = df_train["patient_id"].to_numpy()


# ============================================================
# Grid
# ============================================================

DECODERS = ["argmax", "smooth", "hmm"]
SMOOTH_WINDOWS = [1, 3, 5, 7]
HMM_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]

results = []


# ============================================================
# Grid search
# ============================================================

for decoder in DECODERS:
    for smoothing_window in SMOOTH_WINDOWS:
        # argmax ignores smoothing > 1
        if decoder == "argmax" and smoothing_window != 1:
            continue

        # smooth ignores hmm_alpha
        hmm_alphas = HMM_ALPHAS if decoder == "hmm" else [None]

        for hmm_alpha in hmm_alphas:
            cfg = Config(
                decoder=decoder,
                smoothing_window=smoothing_window,
                hmm_alpha=hmm_alpha if hmm_alpha is not None else 1.0,
            )

            print(f"\n>>> decoder={decoder}, smoothing={smoothing_window}, hmm_alpha={hmm_alpha}")

            scores = cross_validate_xgboost(
                X,
                y,
                df_meta=df_train,
                groups=groups,
                config=cfg,
            )

            mean_f1 = np.mean(scores)
            std_f1 = np.std(scores)

            results.append(
                {
                    "decoder": decoder,
                    "smoothing_window": smoothing_window,
                    "hmm_alpha": hmm_alpha,
                    "mean_f1": mean_f1,
                    "std_f1": std_f1,
                },
            )

            print(f"→ Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")


# ============================================================
# Results table
# ============================================================

df_results = pd.DataFrame(results).sort_values(by="mean_f1", ascending=False)

print("\n========== TOP RESULTS ==========")
print(df_results.head(10))

df_results.to_csv("decoder_grid_results.csv", index=False)
print("\nSaved to decoder_grid_results.csv")

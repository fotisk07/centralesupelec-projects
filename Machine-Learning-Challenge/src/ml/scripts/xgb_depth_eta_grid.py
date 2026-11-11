from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ml.model import cross_validate_xgboost

# ============================================================
# Config (minimal, decoder FROZEN)
# ============================================================


@dataclass
class Config:
    bundle_dir: Path = Path("data")

    # model params (TO TUNE)
    max_depth: int = 6
    eta: float = 0.05

    # model params (FIXED)
    num_classes: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    num_boost_round: int = 500
    seed: int = 42
    n_jobs: int = 4

    # decoder (FROZEN from previous step)
    decoder: str = "hmm"
    hmm_alpha: float = 1.0
    smoothing_window: int = 1

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
# Grid definition
# ============================================================

DEPTHS = [4, 6, 8]
ETAS = [0.03, 0.05, 0.1]

results = []


# ============================================================
# Grid search
# ============================================================

for max_depth, eta in product(DEPTHS, ETAS):
    cfg = Config(
        max_depth=max_depth,
        eta=eta,
    )

    print(f"\n>>> max_depth={max_depth}, eta={eta}")

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
            "max_depth": max_depth,
            "eta": eta,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        },
    )

    print(f"→ Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")


# ============================================================
# Results table
# ============================================================

df_results = pd.DataFrame(results).sort_values("mean_f1", ascending=False)

print("\n========== TOP RESULTS ==========")
print(df_results)

df_results.to_csv("xgb_depth_eta_grid.csv", index=False)
print("\nSaved to xgb_depth_eta_grid.csv")

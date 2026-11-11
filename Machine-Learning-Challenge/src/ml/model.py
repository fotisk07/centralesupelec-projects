from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

Decoder = Literal["argmax", "smooth", "hmm"]


@dataclass
class DecoderContext:
    log_transitions: Optional[np.ndarray] = None
    log_init: Optional[np.ndarray] = None


# ============================================================
# Training
# ============================================================


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config,
) -> xgb.Booster:
    class_counts = Counter(y_train)
    max_count = max(class_counts.values())
    class_weights = {c: (max_count / v) ** config.gamma for c, v in class_counts.items()}
    sample_weights = y_train.map(class_weights).to_numpy()

    dtrain = xgb.DMatrix(
        X_train.values,
        label=y_train.values,
        weight=sample_weights,
        feature_names=X_train.columns.tolist(),
    )

    params = {
        "objective": "multi:softprob",
        "num_class": config.num_classes,
        "eval_metric": "mlogloss",
        "eta": config.eta,
        "max_depth": config.max_depth,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "tree_method": "hist",
        "n_jobs": config.n_jobs,
        "seed": config.seed,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=config.num_boost_round,
        verbose_eval=False,
    )


# ============================================================
# Decoders
# ============================================================


def predict_xgboost_argmax(
    model: xgb.Booster,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Pure epoch-wise argmax decoding.
    """
    dtest = xgb.DMatrix(
        X.values,
        feature_names=X.columns.tolist(),
    )
    probas = model.predict(dtest)
    return np.argmax(probas, axis=1)


def predict_xgboost_smooth(
    model: xgb.Booster,
    X: pd.DataFrame,
    patient_ids: pd.Series,
    epoch_idx: pd.Series,
    *,
    smoothing_window: int,
) -> np.ndarray:
    dtest = xgb.DMatrix(
        X.values,
        feature_names=X.columns.tolist(),
    )

    probas = model.predict(dtest)
    preds = np.zeros(len(X), dtype=int)

    for pid in patient_ids.unique():
        mask = patient_ids == pid
        idx = np.where(mask)[0]

        order = np.argsort(epoch_idx.iloc[idx].values)
        idx_sorted = idx[order]

        p = probas[idx_sorted]

        if smoothing_window > 1:
            p = uniform_filter1d(
                p,
                size=smoothing_window,
                axis=0,
                mode="nearest",
            )

        preds[idx_sorted] = np.argmax(p, axis=1)

    return preds


# ============================================================
# Decoder dispatcher
# ============================================================


def fit_decoder_context(
    y_train: pd.Series,
    df_meta_train: pd.DataFrame,
    config,
) -> DecoderContext:
    if config.decoder == "hmm":
        log_A = estimate_transition_matrix(
            y_train,
            df_meta_train["patient_id"],
            n_classes=config.num_classes,
            alpha=config.hmm_alpha,
        )
        log_pi = estimate_initial_distribution(
            y_train,
            df_meta_train["patient_id"],
            n_classes=config.num_classes,
            alpha=config.hmm_alpha,
        )
        return DecoderContext(
            log_transitions=log_A,
            log_init=log_pi,
        )

    # argmax / smooth need nothing
    return DecoderContext()


def predict_with_decoder(
    model: xgb.Booster,
    X: pd.DataFrame,
    df_meta: pd.DataFrame,
    config,
    ctx: DecoderContext,
) -> np.ndarray:
    if config.decoder == "argmax":
        return predict_xgboost_argmax(model, X)

    if config.decoder == "smooth":
        return predict_xgboost_smooth(
            model,
            X,
            patient_ids=df_meta["patient_id"],
            epoch_idx=df_meta["epoch_idx"],
            smoothing_window=config.smoothing_window,
        )

    if config.decoder == "hmm":
        return predict_xgboost_hmm(
            model,
            X,
            patient_ids=df_meta["patient_id"],
            epoch_idx=df_meta["epoch_idx"],
            log_transitions=ctx.log_transitions,  # ty:ignore[invalid-argument-type]
            log_init=ctx.log_init,  # ty:ignore[invalid-argument-type]
        )

    raise ValueError(f"Unknown decoder: {config.decoder}")


def estimate_transition_matrix(
    y: pd.Series,
    patient_ids: pd.Series,
    n_classes: int,
    alpha: float = 1.0,
) -> np.ndarray:
    A = np.zeros((n_classes, n_classes), dtype=float)

    for pid in patient_ids.unique():
        idx = np.where(patient_ids == pid)[0]
        seq = y.iloc[idx].values

        for a, b in zip(seq[:-1], seq[1:]):
            A[a, b] += 1

    A += alpha
    A /= A.sum(axis=1, keepdims=True)

    return np.log(A)


def estimate_initial_distribution(
    y: pd.Series,
    patient_ids: pd.Series,
    n_classes: int,
    alpha: float = 1.0,
) -> np.ndarray:
    pi = np.zeros(n_classes, dtype=float)

    for pid in patient_ids.unique():
        idx = np.where(patient_ids == pid)[0]
        pi[y.iloc[idx[0]]] += 1

    pi += alpha
    pi /= pi.sum()

    return np.log(pi)


def viterbi_decode(
    log_emissions: np.ndarray,  # (T, K)
    log_transitions: np.ndarray,  # (K, K)
    log_init: np.ndarray,  # (K,)
) -> np.ndarray:
    T, K = log_emissions.shape

    dp = np.zeros((T, K))
    back = np.zeros((T, K), dtype=int)

    dp[0] = log_init + log_emissions[0]

    for t in range(1, T):
        for j in range(K):
            scores = dp[t - 1] + log_transitions[:, j]
            back[t, j] = np.argmax(scores)
            dp[t, j] = scores[back[t, j]] + log_emissions[t, j]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])

    for t in reversed(range(T - 1)):
        path[t] = back[t + 1, path[t + 1]]

    return path


def predict_xgboost_hmm(
    model: xgb.Booster,
    X: pd.DataFrame,
    patient_ids: pd.Series,
    epoch_idx: pd.Series,
    log_transitions: np.ndarray,
    log_init: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    dtest = xgb.DMatrix(
        X.values,
        feature_names=X.columns.tolist(),
    )

    probas = model.predict(dtest)
    preds = np.zeros(len(X), dtype=int)

    for pid in patient_ids.unique():
        mask = patient_ids == pid
        idx = np.where(mask)[0]

        order = np.argsort(epoch_idx.iloc[idx].values)
        idx_sorted = idx[order]

        log_emissions = np.log(probas[idx_sorted] + eps)

        path = viterbi_decode(
            log_emissions,
            log_transitions,
            log_init,
        )

        preds[idx_sorted] = path

    return preds


# ============================================================
# Cross-validation
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold


def cross_validate_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    df_meta: pd.DataFrame,
    groups: np.ndarray,
    config,
):
    gkf = GroupKFold(n_splits=config.cv_n_splits)

    fold_scores = []
    folds = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n--- Fold {fold + 1}/{config.cv_n_splits} ---")

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = train_xgboost(X_tr, y_tr, config)

        ctx = fit_decoder_context(
            y_tr,
            df_meta.iloc[tr_idx],
            config,
        )

        y_pred = predict_with_decoder(
            model,
            X_va,
            df_meta.iloc[va_idx],
            config,
            ctx,
        )

        # ---- metrics ----
        f1 = f1_score(y_va, y_pred, average="macro")
        fold_scores.append(f1)

        print(f"Macro-F1: {f1:.4f}")
        print(
            classification_report(
                y_va,
                y_pred,
                target_names=config.stage_names,
                digits=3,
            )
        )

        # ---- store out-of-fold predictions ----
        folds.append(
            {
                "y_true": y_va.to_numpy(),
                "y_pred": y_pred,
                "patient_id": df_meta.iloc[va_idx]["patient_id"].to_numpy(),
                "epoch_idx": df_meta.iloc[va_idx]["epoch_idx"].to_numpy(),
                "fold": fold,
            }
        )

    print("\n======================")
    print(f"CV Macro-F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    return {
        "fold_scores": fold_scores,
        "folds": folds,
    }

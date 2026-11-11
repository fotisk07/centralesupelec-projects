from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt, hilbert, welch

from ml.data_io import iter_signals_and_targets, iter_signals_only


# -----------------------
# Helpers
# -----------------------
def normalize_per_patient(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


def bandpower(signal, sf, bands):
    f, Pxx = welch(signal, sf, nperseg=512, axis=-1)
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f <= hi)
        out[name] = simpson(Pxx[:, mask], f[mask], axis=-1).mean()
    return out, Pxx


def spectral_entropy(Pxx):
    P = Pxx / (Pxx.sum(axis=-1, keepdims=True) + 1e-12)
    return -np.sum(P * np.log(P + 1e-12), axis=-1).mean()


def hjorth_params(signal):
    first = np.diff(signal, axis=1)
    second = np.diff(first, axis=1)

    var0 = np.var(signal, axis=1)
    var1 = np.var(first, axis=1)
    var2 = np.var(second, axis=1)

    mobility = np.sqrt(var1 / (var0 + 1e-8))
    complexity = np.sqrt(var2 / (var1 + 1e-8)) / (mobility + 1e-8)

    return {
        "hjorth_activity": var0.mean(),
        "hjorth_mobility": mobility.mean(),
        "hjorth_complexity": complexity.mean(),
    }


def spectral_shape(Pxx, f):
    psd = Pxx.mean(axis=0)
    psd = psd / (psd.sum() + 1e-12)

    cumsum = np.cumsum(psd)

    idx50 = min(np.searchsorted(cumsum, 0.5), len(f) - 1)
    idx95 = min(np.searchsorted(cumsum, 0.95), len(f) - 1)

    sef50 = f[idx50]
    sef95 = f[idx95]

    flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)

    return {
        "sef50": sef50,
        "sef95": sef95,
        "spectral_flatness": flatness,
    }


def time_domain_features(epoch):
    x = epoch - epoch.mean(axis=1, keepdims=True)

    zcr = np.mean(np.abs(np.diff(np.sign(x), axis=1)) > 0)
    line_length = np.mean(np.sum(np.abs(np.diff(x, axis=1)), axis=1))
    rms = np.mean(np.sqrt(np.mean(x**2, axis=1)))
    kurt = np.mean(pd.DataFrame(x.T).kurtosis())

    return {
        "zcr": zcr,
        "line_length": line_length,
        "rms": rms,
        "kurtosis": kurt,
    }


def sigma_burst_features(epoch, sf):
    b, a = butter(4, [12, 15], btype="bandpass", fs=sf)
    sig = filtfilt(b, a, epoch, axis=1)

    env = np.abs(hilbert(sig, axis=1))
    env_mean = env.mean(axis=1)
    env_std = env.std(axis=1)

    thresh = env_mean + 2 * env_std
    burst_frac = np.mean(env > thresh[:, None])

    return {
        "sigma_env_mean": env_mean.mean(),
        "sigma_env_std": env_std.mean(),
        "sigma_burst_frac": burst_frac,
    }


def build_feature_df(raw_data, with_labels: bool, config):
    rows = []
    for item in raw_data:
        if with_labels:
            pid, signal, labels = item
        else:
            pid, signal = item

        signal = normalize_per_patient(signal)

        n_epochs = signal.shape[1] // config.samples_per_epoch
        epochs = signal.reshape(signal.shape[0], n_epochs, config.samples_per_epoch)

        for i in range(n_epochs):
            epoch = epochs[:, i, :]

            mean = epoch.mean(axis=1).mean()
            std = epoch.std(axis=1).mean()
            ptp = np.ptp(epoch, axis=1).mean()

            bp_lin, Pxx = bandpower(epoch, config.sf, config.bands)
            f, Pxx = welch(epoch, config.sf, nperseg=512, axis=-1)
            shape_feats = spectral_shape(Pxx, f)
            td_feats = time_domain_features(epoch)
            sigma_feats = sigma_burst_features(epoch, config.sf)
            td_feats.update(sigma_feats)

            total_power = sum(bp_lin.values()) + 1e-8
            bp_rel = {k: v / total_power for k, v in bp_lin.items()}

            bp_log = {k: np.log(v + 1e-8) for k, v in bp_lin.items()}
            bp_rel_log = {f"rel_{k}": np.log(v + 1e-8) for k, v in bp_rel.items()}
            ent = spectral_entropy(Pxx)

            hj = hjorth_params(epoch)

            features = {
                "patient_id": pid,
                "epoch_idx": i,
                "identifier": pid * 10000 + i,
                "mean": mean,
                "std": std,
                "ptp": ptp,
                **bp_log,
                **bp_rel_log,
                **hj,
                **shape_feats,
                **td_feats,
                **sigma_feats,
                "spec_entropy": ent,
                "log_delta_theta": bp_log["delta"] - bp_log["theta"],
                "log_alpha_beta": bp_log["alpha"] - bp_log["beta"],
                "theta_dominance": bp_log["theta"]
                - np.log(np.exp(bp_log["alpha"]) + np.exp(bp_log["beta"]) + 1e-8),
                "beta_silence": bp_log["beta"] - bp_log["theta"],
            }
            if with_labels:
                features["label"] = labels[i]

            rows.append(features)

    return pd.DataFrame(rows)


def add_epoch_drift(
    df: pd.DataFrame,
    *,
    window: int = 2,
    symmetric: bool = True,
) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ["label", "identifier", "patient_id", "epoch_idx"]]

    df = df.copy()
    df["epoch_drift_prev"] = 0.0
    df["epoch_drift_next"] = 0.0

    for pid, block in df.groupby("patient_id", sort=False):
        block = block.sort_values("epoch_idx")  # noqa : PLW2901

        X = block[feature_cols].to_numpy()

        # ---- standardize within patient ----
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # ---- cosine distance ----
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        Xn = X / norms
        dist = 1.0 - np.sum(Xn[1:] * Xn[:-1], axis=1)
        dist = np.concatenate([[0.0], dist])

        n = len(dist)
        drift_prev = np.zeros(n)
        drift_next = np.zeros(n)

        # ---- windowed aggregation ----
        for k in range(1, window + 1):
            drift_prev[k:] += dist[:-k] / window
            drift_next[:-k] += dist[k:] / window

        df.loc[block.index, "epoch_drift_prev"] = drift_prev
        df.loc[block.index, "epoch_drift_next"] = drift_next

    if symmetric:
        df["epoch_drift_mean"] = 0.5 * (df["epoch_drift_prev"] + df["epoch_drift_next"])

    return df


@dataclass
class Config:
    # data params
    bundle_dir: Path = Path("data")
    # feature params
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
    seed: int = 42
    smoothing_window: int = 5
    n_jobs: int = 4


if __name__ == "__main__":
    conf = Config()
    raw_train_data = iter_signals_and_targets(conf.bundle_dir, split="train")
    raw_test_data = iter_signals_only(conf.bundle_dir, split="test")

    df_train = build_feature_df(raw_train_data, with_labels=True, config=conf)
    df_train = add_epoch_drift(
        df_train,
        window=conf.add_epoch_drift_window,
        symmetric=conf.add_epoch_drift_symmetric,
    )
    df_test = build_feature_df(raw_test_data, with_labels=False, config=conf)
    df_test = add_epoch_drift(
        df_test,
        window=conf.add_epoch_drift_window,
        symmetric=conf.add_epoch_drift_symmetric,
    )

    out_dir = Path("features")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_features.csv"
    test_path = out_dir / "test_features.csv"
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Feature DataFrames saved to '{train_path}' and '{test_path}'")

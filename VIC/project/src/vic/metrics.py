from typing import Any, List, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from vic.dataloader import split_with_fixed_test


def get_metrics_vs_train_size(
    model: Any,
    train_sizes: Union[np.ndarray, List[int]],
    data: dict,
    test_idx: dict,
    pool_idx: dict,
    seed: int = 0,
) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    accuracy_scores = {}
    conf_matrices = {}

    for train_size in train_sizes:
        Xtr, ytr, Xte, yte = split_with_fixed_test(data, test_idx, pool_idx, train_size, seed=seed)
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        accuracy_scores[train_size] = accuracy_score(yte, y_pred)
        conf_matrices[train_size] = confusion_matrix(yte, y_pred)

    return accuracy_scores, conf_matrices


def get_average_acc_vs_train_size(
    model: Any,
    train_sizes: Union[np.ndarray, List[int]],
    data: dict,
    test_idx: dict,
    pool_idx: dict,
    n_exp: int = 10,
    seed_master: int = 0,
) -> tuple[dict[int, float], dict[int, float]]:
    master_rng = np.random.default_rng(seed_master)

    avg_accuracy_scores = {}

    for i in tqdm(range(n_exp), desc="Experiments"):
        accuracy_scores = {}
        for train_size in train_sizes:
            seed = master_rng.integers(0, 1e6)  # Generate new seed for each train_size

            Xtr, ytr, Xte, yte = split_with_fixed_test(data, test_idx, pool_idx, train_size, seed=seed)
            model.fit(Xtr, ytr)
            y_pred = model.predict(Xte)
            accuracy_scores[train_size] = accuracy_score(yte, y_pred)

        for train_size, acc in accuracy_scores.items():
            avg_accuracy_scores.setdefault(train_size, []).append(acc)

    mean_accuracy_scores = {size: np.mean(accs) for size, accs in avg_accuracy_scores.items()}
    std_accuracy_scores = {size: np.std(accs) for size, accs in avg_accuracy_scores.items()}

    return mean_accuracy_scores, std_accuracy_scores


def get_metrics_vs_train_size_joblib(
    model_factory: callable,
    train_sizes: Union[np.ndarray, List[int]],
    data: dict,
    test_idx: dict,
    pool_idx: dict,
    seed: int = 0,
    n_jobs: int = 4,
):
    def worker(train_size):
        model = model_factory()
        Xtr, ytr, Xte, yte = split_with_fixed_test(data, test_idx, pool_idx, train_size, seed=seed)
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        return train_size, accuracy_score(yte, y_pred), confusion_matrix(yte, y_pred)

    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(worker)(ts) for ts in train_sizes)

    acc = {ts: a for ts, a, _ in results}
    cms = {ts: cm for ts, _, cm in results}
    return acc, cms

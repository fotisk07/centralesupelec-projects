import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from vic.dataloader import load_yale, make_fixed_test_indices, split_with_fixed_test
from vic.models import LBPH, DeepLearningModel, Eigenfaces, Fisherfaces


def evaluate_models_vs_train_size(
    models: dict,  # {"Eigenfaces": model, "LBPH": model}
    train_sizes: list[int],
    data: dict,
    test_idx: dict,
    pool_idx: dict,
    n_exp: int = 10,
    seed_master: int = 0,
    save_path: str = "logs/results.json",
):
    # -------------------------
    # Load previous results if they exist
    # -------------------------
    save_path = Path(save_path)
    if os.path.exists(save_path):
        with open(save_path) as f:
            results = json.load(f)
        print(f"[INFO] Loaded existing results from {save_path}")
    else:
        results = {}

    rng = np.random.default_rng(seed_master)
    for model_name, model_fn in models.items():
        print(f"\n=== Model: {model_name} ===")

        results.setdefault(model_name, {})

        for train_size in train_sizes:
            results[model_name].setdefault(str(train_size), [])

            existing_runs = len(results[model_name][str(train_size)])
            remaining = n_exp - existing_runs

            if remaining <= 0:
                print(f"  [SKIP] train_size={train_size} already done")
                continue

            print(f"  train_size={train_size} ({existing_runs}/{n_exp} done)")

            for _ in tqdm(range(remaining), desc="    runs", leave=False):
                seed = int(rng.integers(0, 1e6))

                Xtr, ytr, Xte, yte = split_with_fixed_test(
                    data,
                    test_idx,
                    pool_idx,
                    train_size,
                    seed=seed,
                )
                model = model_fn()

                # Fisherfaces needs more training samples than class
                if isinstance(model, Fisherfaces) and Xtr.shape[0] <= np.unique(ytr).size:
                    print(
                        "  [SKIP] Fisherfaces invalid for this split: "
                        f"n_samples={Xtr.shape[0]} <= n_classes={np.unique(ytr).size}"
                    )
                    if len(results[model_name][str(train_size)]) == 0:
                        results[model_name].pop(str(train_size), None)
                    break

                model.fit(Xtr, ytr)
                y_pred = model.predict(Xte)

                acc = accuracy_score(yte, y_pred)
                results[model_name][str(train_size)].append(acc)

                with open(save_path, "w") as f:
                    json.dump(results, f, indent=2)

    return results


data = load_yale("data/Yale")
test_idx, pool_idx = make_fixed_test_indices(data, n_test=3, seed=0)
eigen_faces = Eigenfaces(K=10)
lbph = LBPH()

models = {
    "Eigenfaces": lambda: Eigenfaces(K=30),
    "LBPH": lambda: LBPH(),
    "Fisherfaces": lambda: Fisherfaces(strict=False, normalize=False, random_state=0),
    "Deep Learning": lambda: DeepLearningModel(),
}

train_sizes = [2, 3, 4, 5, 6, 7, 8]

results = evaluate_models_vs_train_size(
    models=models,
    train_sizes=train_sizes,
    data=data,
    test_idx=test_idx,
    pool_idx=pool_idx,
    n_exp=3,
    save_path="logs/yale_learning_curves.json",
)

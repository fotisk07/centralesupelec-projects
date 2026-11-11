import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_orl(path: str = "data/ORL") -> dict[int, np.ndarray]:
    p = Path(path)
    data = {}

    ids = sorted(int(a.name[1:]) for a in p.glob("s*") if a.is_dir() and a.name[1:].isdigit())

    for sid in ids:
        subject_dir = p / f"s{sid}"
        imgs = []
        for img_path in sorted(subject_dir.glob("*.pgm")):
            img = Image.open(img_path)
            imgs.append(np.asarray(img, dtype=np.float32))
        data[sid] = np.stack(imgs)  # (n_i, H, W)

    return data


def load_yale(path: str = "data/Yale") -> dict[int, np.ndarray]:
    p = Path(path)
    data = {}
    temp_data = defaultdict(list)

    files = sorted(x for x in p.iterdir() if x.is_file())

    for file_path in files:
        match = re.match(r"subject(\d+)", file_path.name)
        if not match:
            continue

        sid = int(match.group(1))

        try:
            img = Image.open(file_path).convert("L")  # enforce grayscale
            img_arr = np.asarray(img, dtype=np.float32)
            temp_data[sid].append(img_arr)
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}") from e

    for sid, imgs in temp_data.items():
        data[sid] = np.stack(imgs)

    return data


def make_fixed_test_indices(data: dict, n_test: int = 5, seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    test_idx = {}
    pool_idx = {}
    for sid, imgs in data.items():
        idx = rng.permutation(len(imgs))
        test_idx[sid] = idx[:n_test]
        pool_idx[sid] = idx[n_test:]
    return test_idx, pool_idx


def split_with_fixed_test(
    data: dict,
    test_idx: dict,
    pool_idx: dict,
    n_train: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = [], [], [], []

    for sid, imgs in data.items():
        # fixed test
        te = test_idx[sid]
        Xte.append(imgs[te])
        yte.append(np.full(len(te), sid))

        # variable train sampled from remaining pool
        pool = pool_idx[sid]
        tr = rng.choice(pool, size=n_train, replace=False)
        Xtr.append(imgs[tr])
        ytr.append(np.full(len(tr), sid))

    return (
        np.concatenate(Xtr),
        np.concatenate(ytr),
        np.concatenate(Xte),
        np.concatenate(yte),
    )

import os
import time
from pathlib import Path

import numpy as np

CHECKPOINT_EVERY = 50
LGBM_CONFIGS = {
    "default": {},
    "small-leaf": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 5,
                   "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8},
}


def save_scores(path, S, done, seconds):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, S=S, done=done, seconds=seconds)
    os.replace(tmp, path)


def tabpfn_per_label(X_tr, Y_tr, X_te, scored, device="auto", checkpoint=None, shard=(0, 1)):
    os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")
    from tabpfn import TabPFNClassifier

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    S = np.zeros((len(X_te), Y_tr.shape[1]))
    done = np.zeros(Y_tr.shape[1], dtype=bool)
    spent = 0.0
    if checkpoint is not None and Path(checkpoint).exists():
        with np.load(checkpoint) as saved:
            S, done, spent = saved["S"], saved["done"], float(saved["seconds"])
    start = time.time()
    todo = np.flatnonzero(scored)[shard[0]::shard[1]]
    clf = TabPFNClassifier(device=device)
    for k, j in enumerate(todo[~done[todo]]):
        pos = int(Y_tr[:, j].sum())
        if pos == 0 or pos == len(Y_tr):
            S[:, j] = float(pos > 0)
        else:
            clf.fit(X_tr, Y_tr[:, j].astype(int))
            S[:, j] = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
        done[j] = True
        if checkpoint is not None and k % CHECKPOINT_EVERY == CHECKPOINT_EVERY - 1:
            save_scores(checkpoint, S, done, spent + time.time() - start)
    if checkpoint is not None:
        save_scores(checkpoint, S, done, spent + time.time() - start)
    return S, done


def _lgbm_column(X_tr, y, X_te, params):
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(random_state=0, n_jobs=1, verbose=-1, **params)
    clf.fit(X_tr, y)
    return clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]


def lgbm_per_label(X_tr, Y_tr, X_te, scored, params, n_jobs=-1):
    from joblib import Parallel, delayed

    S = np.zeros((len(X_te), Y_tr.shape[1]))
    pos = Y_tr.sum(0)
    classes = np.flatnonzero(scored)
    both = (pos[classes] > 0) & (pos[classes] < len(Y_tr))
    S[:, classes[~both]] = (pos[classes[~both]] > 0).astype(float)
    fit = classes[both]
    cols = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(_lgbm_column)(X_tr, Y_tr[:, j].astype(int), X_te, params) for j in fit)
    if cols:
        S[:, fit] = np.column_stack(cols)
    return S

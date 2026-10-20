import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data import DATA_DIR, TREES, dataset_of, load, load_selection
from .hierarchy import max_constraint, min_constraint, projection, tree_map, violations
from .metrics import auprc, brier, exact_match, hamming_score
from .models import LGBM_CONFIGS, lgbm_per_label, tabpfn_per_label


def update_csv(rows, path):
    path = Path(path)
    new = pd.DataFrame(rows)
    if new.empty:
        return
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["dataset"].isin(new["dataset"])], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


def fit(datasets, scores_dir, data_dir=DATA_DIR, device="auto", shard=(0, 1)):
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        d = load(name, data_dir)
        path = scores_dir / f"{name}-{shard[0]}of{shard[1]}.npz"
        tabpfn_per_label(d["X_tr"], d["Y_tr"], d["X_te"], d["eval"], device, checkpoint=path, shard=shard)
        print(f"{name}: shard {shard[0]} of {shard[1]} done", flush=True)


def load_scores(name, shape, scores_dir):
    S, done, seconds = np.zeros(shape), np.zeros(shape[1], dtype=bool), 0.0
    for path in sorted(Path(scores_dir).glob(f"{name}-*of*.npz")):
        with np.load(path) as z:
            S[:, z["done"]] = z["S"][:, z["done"]]
            done |= z["done"]
            seconds += float(z["seconds"])
    return S, done, seconds


def evaluate(datasets, scores_dir, out_dir, data_dir=DATA_DIR):
    decoders, sets, fits = [], [], []
    for name in datasets:
        d = load(name, data_dir)
        ev, Y, par = d["eval"], d["Y_te"], d["par"]
        S, done, seconds = load_scores(name, (len(Y), len(d["names"])), scores_dir)
        if not done[ev].all():
            print(f"{name}: {int((~done[ev]).sum())} scored classes have no scores yet, skipped", flush=True)
            continue
        start = time.time()
        iso, sweeps = projection(S, par, ev)
        iso_seconds = time.time() - start
        mcm = max_constraint(S, par)
        for decoder, M in (("raw", S), ("MCM", mcm), ("MIN", min_constraint(S, par, ev)), ("ISO", iso)):
            decoders.append({"dataset": name, "decoder": decoder, "auprc": auprc(Y, M, ev), "brier": brier(Y, M, ev),
                             "f1_at_05": float(f1_score(Y[:, ev].ravel(), (M[:, ev] >= 0.5).ravel())),
                             "violating_rows": violations(M, par, ev)})
        pos = d["Y_tr"][:, ev].sum(0)
        fits.append({"dataset": name, "rows_train": len(d["Y_tr"]), "rows_test": len(Y),
                     "features": d["X_tr"].shape[1], "classes": int(ev.sum()),
                     "fitted": int(((pos > 0) & (pos < len(d["Y_tr"]))).sum()), "seconds": round(seconds, 1),
                     "iso_sweeps": sweeps, "iso_seconds": round(iso_seconds, 1)})
        if name in TREES:
            labelled = {"THR": (iso >= 0.5).astype(np.int8), "MAP": tree_map(iso, par, ev),
                        "MCM": (mcm >= 0.5).astype(np.int8)}
            for decoder, P in labelled.items():
                sets.append({"dataset": name, "decoder": decoder, "exact_match": exact_match(Y, P, ev),
                             "hamming": hamming_score(Y, P, ev), "violating_rows": violations(P, par, ev),
                             "differs_from_thr": float((P[:, ev] != labelled["THR"][:, ev]).any(axis=1).mean())})
        print(f"{name}: evaluated", flush=True)
    out_dir = Path(out_dir)
    update_csv(decoders, out_dir / "decoders.csv")
    update_csv(sets, out_dir / "label_sets.csv")
    update_csv(fits, out_dir / "fits.csv")


def lgbm(datasets, out, data_dir=DATA_DIR, n_jobs=-1):
    for name in datasets:
        sel = load_selection(name, data_dir)
        validation = {}
        for config, params in LGBM_CONFIGS.items():
            S = lgbm_per_label(sel["X_tr"], sel["Y_tr"], sel["X_te"], sel["eval"], params, n_jobs)
            validation[config] = round(auprc(sel["Y_te"], max_constraint(S, sel["par"]), sel["eval"]), 4)
        chosen = max(validation, key=validation.get)
        d = load(name, data_dir)
        S = lgbm_per_label(d["X_tr"], d["Y_tr"], d["X_te"], d["eval"], LGBM_CONFIGS[chosen], n_jobs)
        update_csv([{"dataset": name, "config": chosen, "validation": json.dumps(validation),
                     "auprc_raw": auprc(d["Y_te"], S, d["eval"]),
                     "auprc_mcm": auprc(d["Y_te"], max_constraint(S, d["par"]), d["eval"])}], out)
        print(f"{name}: {chosen} {validation}", flush=True)


def read_reruns(folder, method):
    rows = []
    for path in sorted(Path(folder).glob("*.csv")):
        name = dataset_of(path.stem)
        if name is None:
            continue
        d = pd.read_csv(path, header=None, names=["seed", "epoch", "auprc"])
        for seed, g in d.groupby("seed"):
            last = g.loc[g["epoch"].idxmax()]
            rows.append({"dataset": name, "method": method, "seed": int(seed), "epochs": int(last["epoch"]) + 1,
                         "final": float(last["auprc"]), "best_on_test": float(g["auprc"].max())})
    return rows


def reruns(chmcnn, spur, out):
    rows = [r for f in chmcnn for r in read_reruns(f, "C-HMCNN")] + [r for f in spur for r in read_reruns(f, "SPUR")]
    table = pd.DataFrame(rows).drop_duplicates(["dataset", "method", "seed"], keep="last")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"{len(table)} runs from {table['dataset'].nunique()} datasets", flush=True)

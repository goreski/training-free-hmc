import os
from pathlib import Path

import numpy as np

from .hierarchy import ancestors

DATA_DIR = Path(os.environ.get("TFHMC_DATA", "data"))
FUNCAT = ["cellcycle", "derisi", "eisen", "expr", "gasch1", "gasch2", "seq", "spo"]
GO = [f"{d}_GO" for d in FUNCAT]
OTHER_FILES = {
    "diatoms": ("Diatoms_train", "Diatoms_test"),
    "enron": ("Enron_corr_trainvalid", "Enron_corr_test"),
    "imclef07a": ("ImCLEF07A_Train", "ImCLEF07A_Test"),
    "imclef07d": ("ImCLEF07D_Train", "ImCLEF07D_Test"),
}
OTHERS = list(OTHER_FILES)
DATASETS = FUNCAT + GO + OTHERS
TREES = FUNCAT + OTHERS
UNSCORED = {"root", "GO0003674", "GO0005575", "GO0008150"}
SPLITS = ("train", "valid", "test")


def parse_arff(path, is_go=False):
    parsers, names, rows, labels = [], None, [], []
    reading = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("@ATTRIBUTE"):
                if line.startswith("@ATTRIBUTE class"):
                    branches = line.split("hierarchical", 1)[1].strip().split(",")
                    up = {"root": set()}
                    for branch in branches:
                        terms = branch.strip().split("/")
                        if is_go:
                            up.setdefault(terms[0], set())
                            up.setdefault(terms[1], set()).add(terms[0])
                        else:
                            for i in range(1, len(terms) + 1):
                                up.setdefault(".".join(terms[:i]), set()).add(".".join(terms[:i - 1]) or "root")
                    names = sorted(up, key=lambda n: (0, "") if n == "root" else (len(n.split(".")), n))
                    index = {n: i for i, n in enumerate(names)}
                    par = [sorted(index[p] for p in up[n]) for n in names]
                    closure = ancestors(par)
                else:
                    _, _, ftype = line.split(None, 2)
                    ftype = ftype.strip()
                    if ftype.lower() == "numeric":
                        parsers.append(("num", None))
                    else:
                        parsers.append(("cat", [c.strip() for c in ftype.strip("{}").split(",")]))
            elif line.startswith("@DATA"):
                reading = True
            elif reading and line.strip():
                values = line.split("%")[0].strip().split(",")
                x = []
                for (kind, cats), v in zip(parsers, values[: len(parsers)]):
                    v = v.strip()
                    if kind == "num":
                        x.append(float(v) if v != "?" else np.nan)
                    else:
                        x.extend(float(v == c) for c in cats)
                y = np.zeros(len(names))
                y[index["root"]] = 1
                for term in values[len(parsers)].strip().split("@"):
                    y[closure[index[term.strip().replace("/", ".")]]] = 1
                rows.append(x)
                labels.append(y)
    X = np.array(rows, dtype=float)
    mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), np.where(np.isnan(mean), 0.0, mean), X)
    return {"X": X, "Y": np.stack(labels).astype(np.int8), "names": names, "par": par}


def files(name, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    if name.endswith("_GO"):
        return [data_dir / "datasets_GO" / name / f"{name}.{s}.arff" for s in SPLITS], True
    if name in OTHER_FILES:
        return [data_dir / "others" / f"{f}.arff" for f in OTHER_FILES[name]], False
    return [data_dir / "datasets_FUN" / f"{name}_FUN" / f"{name}_FUN.{s}.arff" for s in SPLITS], False


def load(name, data_dir=DATA_DIR):
    paths, is_go = files(name, data_dir)
    parts = [parse_arff(p, is_go) for p in paths]
    names = parts[0]["names"]
    if not all(p["names"] == names and p["par"] == parts[0]["par"] for p in parts):
        raise ValueError(f"{name}: the files have different hierarchies")
    return {"X_tr": np.vstack([p["X"] for p in parts[:-1]]), "Y_tr": np.vstack([p["Y"] for p in parts[:-1]]),
            "X_te": parts[-1]["X"], "Y_te": parts[-1]["Y"], "names": names, "par": parts[0]["par"],
            "eval": np.array([n not in UNSCORED for n in names])}


def load_selection(name, data_dir=DATA_DIR):
    paths, is_go = files(name, data_dir)
    parts = [parse_arff(p, is_go) for p in paths[:-1]]
    if len(parts) == 2:
        fit, val = parts
        X_fit, Y_fit, X_val, Y_val = fit["X"], fit["Y"], val["X"], val["Y"]
    else:
        X, Y = parts[0]["X"], parts[0]["Y"]
        order = np.random.default_rng(0).permutation(len(Y))
        val, fit = order[: len(Y) // 4], order[len(Y) // 4:]
        X_fit, Y_fit, X_val, Y_val = X[fit], Y[fit], X[val], Y[val]
    names = parts[0]["names"]
    return {"X_tr": X_fit, "Y_tr": Y_fit, "X_te": X_val, "Y_te": Y_val, "names": names, "par": parts[0]["par"],
            "eval": np.array([n not in UNSCORED for n in names])}


def dataset_of(results_name):
    for suffix, keep in (("_FUN", False), ("_GO", True), ("_others", False)):
        if results_name.endswith(suffix):
            return results_name if keep else results_name[: -len(suffix)]
    return None

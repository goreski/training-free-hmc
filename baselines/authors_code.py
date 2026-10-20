import os
import runpy
import sys
import types

import numpy as np


def install_shims():
    import networkx as nx
    import scipy.sparse as sp
    import sklearn.metrics as skm

    if not hasattr(nx, "to_numpy_matrix"):
        def to_numpy_matrix(G, nodelist=None, dtype=None, order=None, multigraph_weight=sum, weight="weight",
                            nonedge=0.0):
            order = order if isinstance(order, str) else None
            return np.asmatrix(nx.to_numpy_array(G, nodelist=nodelist, dtype=dtype, order=order,
                                                 multigraph_weight=multigraph_weight, weight=weight, nonedge=nonedge))
        nx.to_numpy_matrix = to_numpy_matrix
    if not hasattr(nx, "to_scipy_sparse_matrix"):
        def to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight="weight", format="csr"):
            matrix = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)
            return sp.csr_matrix(matrix).asformat(format)
        nx.to_scipy_sparse_matrix = to_scipy_sparse_matrix
    if not hasattr(skm, "jaccard_similarity_score"):
        skm.jaccard_similarity_score = skm.jaccard_score
    try:
        import keras
    except ImportError:
        def to_categorical(y, num_classes=None, dtype="float32"):
            y = np.asarray(y, dtype=int)
            n = int(num_classes) if num_classes is not None else int(y.max()) + 1
            out = np.zeros(y.shape + (n,), dtype=dtype)
            np.put_along_axis(out, y[..., None], 1, axis=-1)
            return out
        keras = types.ModuleType("keras")
        keras.utils = types.ModuleType("keras.utils")
        keras.utils.to_categorical = to_categorical
        sys.modules["keras"], sys.modules["keras.utils"] = keras, keras.utils


def main():
    repo, dataset, seed = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
    install_shims()
    os.chdir(repo)
    sys.path.insert(0, repo)
    gin = os.path.join(repo, "modules", "gin")
    if os.path.isdir(gin):
        sys.path.append(gin)
    os.makedirs("results", exist_ok=True)
    sys.argv = ["main.py", "--dataset", dataset, "--seed", str(seed), "--device", "0"]
    runpy.run_path("main.py", run_name="__main__")


if __name__ == "__main__":
    main()

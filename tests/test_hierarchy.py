import itertools

import numpy as np
import pytest
from scipy.optimize import minimize

from tfhmc.hierarchy import children_first, max_constraint, min_constraint, projection, tree_map, violations

DAG = [[], [5], [5], [1, 2], [3], [0]]
DAG_SCORED = np.array([False, True, True, True, True, False])
TREE = [[], [0], [0], [1], [1], [2], [5]]
TREE_SCORED = np.array([False] + [True] * 6)


def test_max_constraint_and_violations():
    S = np.array([[0, 0.1, 0.2, 0.3, 0.9, 0.0], [0, 0.2, 0.7, 0.5, 0.1, 0.0]])
    M = max_constraint(S, DAG)
    np.testing.assert_allclose(M, [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [0.7, 0.5, 0.7, 0.5, 0.1, 0.7]])
    assert violations(M, DAG, DAG_SCORED) == 0.0
    assert violations(np.array([[0, 0.7, 0.4, 0.6, 0, 0]]), DAG, DAG_SCORED) == 1.0
    assert violations(np.array([[0, 0.9, 0.0, 0.0, 0, 0]]), DAG, DAG_SCORED) == 0.0


def test_min_constraint():
    S = np.array([[0, 0.9, 0.2, 0.5, 0.7, 0.0]])
    np.testing.assert_allclose(min_constraint(S, DAG, DAG_SCORED), [[0, 0.9, 0.2, 0.2, 0.2, 0.0]])


def test_projection_is_the_nearest_consistent_vector():
    edges = [(c, p) for c, ps in enumerate(DAG) for p in ps if DAG_SCORED[c] and DAG_SCORED[p]]
    S = np.random.default_rng(1).random((6, 6))
    X, sweeps = projection(S, DAG, DAG_SCORED, tol=1e-10)
    assert sweeps > 1
    for row, x in zip(S, X):
        cons = [{"type": "ineq", "fun": lambda z, c=c, p=p: z[p] - z[c]} for c, p in edges]
        ref = minimize(lambda z: ((z - row) ** 2).sum(), row, constraints=cons, method="SLSQP", tol=1e-12).x
        np.testing.assert_allclose(x[DAG_SCORED], ref[DAG_SCORED], atol=1e-6)
        assert x[0] == row[0] and x[5] == row[5]
    assert violations(X, DAG, DAG_SCORED) == 0.0
    consistent = np.array([[0, 0.9, 0.8, 0.7, 0.1, 0]])
    np.testing.assert_allclose(projection(consistent, DAG, DAG_SCORED)[0], consistent)
    one = np.array([[0, 0.9, 0.9, 0.2, 0.6, 0]])
    np.testing.assert_allclose(projection(one, DAG, DAG_SCORED, tol=1e-10)[0], [[0, 0.9, 0.9, 0.4, 0.4, 0]])


def test_cycle_is_rejected():
    with pytest.raises(ValueError):
        children_first([[1], [0]])


def _tree_prob(x, s):
    p = 1.0
    for c, ps in enumerate(TREE):
        if not TREE_SCORED[c]:
            continue
        up = ps[0]
        if TREE_SCORED[up] and not x[up]:
            if x[c]:
                return 0.0
            continue
        base = s[up] if TREE_SCORED[up] else 1.0
        q = s[c] / base if base > 0 else 0.0
        p *= q if x[c] else 1 - q
    return p


SETS = [np.array((1,) + bits) for bits in itertools.product((0, 1), repeat=6)]


def test_tree_map_is_the_most_probable_consistent_set():
    q = np.random.default_rng(3).random((40, 7))
    q[0, 1], q[1, 5], q[2, 2], q[3, 3] = 0.0, 1.0, 0.0, 1.0
    S = np.zeros_like(q)
    for c in reversed(children_first(TREE)):
        if TREE_SCORED[c]:
            up = TREE[c][0]
            S[:, c] = q[:, c] * (S[:, up] if TREE_SCORED[up] else 1.0)
    P = tree_map(S, TREE, TREE_SCORED)
    assert violations(P, TREE, TREE_SCORED) == 0.0 and (P[:, 0] == 1).all()
    for s, x in zip(S, P):
        probs = [_tree_prob(z, s) for z in SETS]
        assert abs(sum(probs) - 1) < 1e-12
        np.testing.assert_allclose(sum(p * z for p, z in zip(probs, SETS))[1:], s[1:], atol=1e-12)
        assert _tree_prob(x, s) >= max(probs) - 1e-12


def test_tree_map_ties_and_edge_cases():
    S = np.array([[0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0],
                  [0, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0],
                  [0, 0.6, 0.0, 0.35, 0.35, 0.0, 0.0]])
    P = tree_map(S, TREE, TREE_SCORED)
    np.testing.assert_array_equal(P, [[1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(P[2], max(SETS, key=lambda z: _tree_prob(z, S[2])))
    with pytest.raises(ValueError):
        tree_map(S[:, :5], [[], [0], [0], [1], [1, 2]], TREE_SCORED[:5])

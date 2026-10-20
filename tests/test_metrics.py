import numpy as np
import pytest
from sklearn.metrics import accuracy_score, average_precision_score, hamming_loss

from tfhmc.metrics import auprc, exact_match, hamming_score

SCORED = np.array([False, True, True, True, True, True, True])


def test_auprc_is_micro_over_scored_classes():
    rng = np.random.default_rng(0)
    Y, S = rng.integers(0, 2, (50, 7)), rng.random((50, 7))
    assert auprc(Y, S, SCORED) == pytest.approx(average_precision_score(Y[:, 1:], S[:, 1:], average="micro"))


def test_set_metrics():
    rng = np.random.default_rng(0)
    Y, P = (rng.random((50, 7)) < 0.3).astype(np.int8), (rng.random((50, 7)) < 0.3).astype(np.int8)
    P[:10] = Y[:10]
    P[:5, 0] = 1 - Y[:5, 0]
    assert exact_match(Y, P, SCORED) == accuracy_score(Y[:, SCORED], P[:, SCORED]) >= 0.2
    assert hamming_score(Y, P, SCORED) == pytest.approx(1 - hamming_loss(Y[:, SCORED], P[:, SCORED]))

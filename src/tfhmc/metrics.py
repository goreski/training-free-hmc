from sklearn.metrics import average_precision_score


def auprc(Y, S, scored):
    return float(average_precision_score(Y[:, scored], S[:, scored], average="micro"))


def exact_match(Y, P, scored):
    return float((P[:, scored] == Y[:, scored]).all(axis=1).mean())


def hamming_score(Y, P, scored):
    return float(1 - (P[:, scored] != Y[:, scored]).mean())


def brier(Y, S, scored):
    return float(((S[:, scored] - Y[:, scored]) ** 2).mean())

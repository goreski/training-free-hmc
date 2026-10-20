import numpy as np


def children_first(par):
    waiting = np.zeros(len(par), dtype=int)
    for ps in par:
        for p in ps:
            waiting[p] += 1
    order, ready = [], [i for i in range(len(par)) if waiting[i] == 0]
    while ready:
        c = ready.pop()
        order.append(c)
        for p in par[c]:
            waiting[p] -= 1
            if waiting[p] == 0:
                ready.append(p)
    if len(order) != len(par):
        raise ValueError("the hierarchy has a cycle")
    return order


def ancestors(par):
    up = [set() for _ in par]
    for i in reversed(children_first(par)):
        up[i] = {i}.union(*(up[p] for p in par[i]))
    return [np.array(sorted(a)) for a in up]


def scored_edges(par, scored):
    edges = [(c, p) for c, ps in enumerate(par) for p in ps if scored[c] and scored[p]]
    return np.array(edges, dtype=int).reshape(-1, 2)


def max_constraint(S, par):
    S = S.copy()
    for c in children_first(par):
        for p in par[c]:
            np.maximum(S[:, p], S[:, c], out=S[:, p])
    return S


def min_constraint(S, par, scored):
    S = S.copy()
    for c in reversed(children_first(par)):
        if scored[c]:
            for p in par[c]:
                if scored[p]:
                    np.minimum(S[:, c], S[:, p], out=S[:, c])
    return S


def _disjoint_batches(edges):
    batches, members = [], []
    for e, (c, p) in enumerate(edges):
        for b, nodes in enumerate(members):
            if c not in nodes and p not in nodes:
                batches[b].append(e)
                nodes.update((c, p))
                break
        else:
            batches.append([e])
            members.append({c, p})
    return [np.array(b) for b in batches]


def projection(S, par, scored, tol=1e-7, max_sweeps=20000):
    edges = scored_edges(par, scored)
    x = S.astype(float).copy()
    if len(edges) == 0:
        return x, 0
    batches = _disjoint_batches(edges)
    qc, qp = np.zeros((len(x), len(edges))), np.zeros((len(x), len(edges)))
    for sweep in range(1, max_sweeps + 1):
        before = x.copy()
        for b in batches:
            c, p = edges[b, 0], edges[b, 1]
            yc, yp = x[:, c] + qc[:, b], x[:, p] + qp[:, b]
            over = yc > yp
            mid = 0.5 * (yc + yp)
            zc, zp = np.where(over, mid, yc), np.where(over, mid, yp)
            qc[:, b], qp[:, b] = yc - zc, yp - zp
            x[:, c], x[:, p] = zc, zp
        if np.abs(x - before).max() < tol and (x[:, edges[:, 0]] - x[:, edges[:, 1]]).max() < tol:
            break
    scored_par = [[p for p in ps if scored[c] and scored[p]] for c, ps in enumerate(par)]
    return max_constraint(x, scored_par), sweep


def violations(S, par, scored, threshold=0.5):
    edges = scored_edges(par, scored)
    if len(edges) == 0:
        return 0.0
    bad = (S[:, edges[:, 0]] >= threshold) & (S[:, edges[:, 1]] < threshold)
    return float(bad.any(axis=1).mean())


def tree_map(S, par, scored):
    if not all(len(ps) == 1 for c, ps in enumerate(par) if scored[c]):
        raise ValueError("MAP decoding needs a tree")
    if not all(not ps for c, ps in enumerate(par) if not scored[c]):
        raise ValueError("only roots may be unscored")
    n = len(par)
    up = np.array([ps[0] if ps else 0 for ps in par])
    top = np.array([not scored[up[c]] for c in range(n)])
    parent = np.where(top, 1.0, S[:, up])
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.clip(np.where(parent > 0, S / parent, 0.0), 0.0, 1.0)
        log_in, log_out = np.log(q), np.log1p(-q)
    best = np.zeros(S.shape)
    keep = np.zeros(S.shape, dtype=bool)
    order = children_first(par)
    for c in order:
        if scored[c]:
            gain = log_in[:, c] + best[:, c]
            keep[:, c] = gain >= log_out[:, c]
            best[:, up[c]] += np.where(keep[:, c], gain, log_out[:, c])
    present = np.zeros(S.shape, dtype=np.int8)
    for c in reversed(order):
        present[:, c] = keep[:, c] & (present[:, up[c]] == 1) if scored[c] else 1
    return present

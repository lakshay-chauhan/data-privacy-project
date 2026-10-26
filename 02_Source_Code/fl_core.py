"""
fl_core.py - Core library for the privacy-aware cloud-edge federated analytics testbed.

Implements (NumPy only, fully reproducible):
  * Random-Fourier-feature softmax classifier (edge-friendly model)
  * Non-IID Dirichlet partitioning across clients
  * Client-side DP-SGD (per-example clipping + Gaussian noise, Poisson sampling)
  * Renyi-DP accountant for the sampled Gaussian mechanism (integer orders)
  * FedAvg aggregation, coordinate-wise median aggregation
  * Additive-mask secure aggregation simulation (pairwise masks cancel)
  * Loss-threshold membership inference attack (AUC)
"""
import math
import time
import numpy as np
from scipy.special import logsumexp, gammaln
from sklearn.metrics import roc_auc_score, f1_score


# ----------------------------------------------------------------------------
# Model: random Fourier features + softmax regression
# ----------------------------------------------------------------------------
class RFFMap:
    def __init__(self, d_in, d_out=256, gamma=None, seed=0):
        rng = np.random.RandomState(seed)
        gamma = gamma if gamma is not None else 1.0 / d_in
        self.W = rng.normal(0.0, math.sqrt(2.0 * gamma), size=(d_in, d_out))
        self.b = rng.uniform(0, 2 * math.pi, size=d_out)
        self.d_out = d_out

    def __call__(self, X):
        Z = math.sqrt(2.0 / self.d_out) * np.cos(X @ self.W + self.b)
        return np.hstack([Z, np.ones((Z.shape[0], 1))])  # bias column


def softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def predict(W, Z):
    return np.argmax(Z @ W, axis=1)


def accuracy(W, Z, y):
    return float((predict(W, Z) == y).mean())


def per_example_loss(W, Z, y):
    P = softmax(Z @ W)
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))


# ----------------------------------------------------------------------------
# Partitioning
# ----------------------------------------------------------------------------
def dirichlet_partition(y_train, y_test, n_clients, alpha, seed):
    """Partition train and test indices with the same class-to-client proportions."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    tr = [[] for _ in range(n_clients)]
    te = [[] for _ in range(n_clients)]
    for c in classes:
        p = rng.dirichlet(alpha * np.ones(n_clients))
        for arr, y, store in ((np.where(y_train == c)[0], y_train, tr),
                              (np.where(y_test == c)[0], y_test, te)):
            rng.shuffle(arr)
            cuts = (np.cumsum(p) * len(arr)).astype(int)[:-1]
            for k, part in enumerate(np.split(arr, cuts)):
                store[k].extend(part.tolist())
    # guard: every client has at least 8 train samples
    for k in range(n_clients):
        while len(tr[k]) < 8:
            donor = int(np.argmax([len(t) for t in tr]))
            tr[k].append(tr[donor].pop())
    return [np.array(t) for t in tr], [np.array(t) for t in te]


# ----------------------------------------------------------------------------
# Renyi DP accountant for the sampled Gaussian mechanism (Mironov et al., 2019)
# ----------------------------------------------------------------------------
def _log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rdp_sampled_gaussian(q, sigma, alpha):
    """RDP of order alpha (integer >= 2) for one step with Poisson sampling rate q."""
    if q <= 0:
        return 0.0
    if sigma <= 0:
        return float("inf")
    if q >= 1.0:
        return alpha / (2.0 * sigma ** 2)
    ks = np.arange(0, alpha + 1)
    logterms = (_log_comb(alpha, ks) + (alpha - ks) * math.log(1 - q)
                + ks * math.log(q) + (ks * ks - ks) / (2.0 * sigma ** 2))
    return float(logsumexp(logterms) / (alpha - 1))


def epsilon_from_rdp(q, sigma, steps, delta=1e-5, orders=range(2, 129)):
    if sigma <= 0:
        return float("inf")
    best = float("inf")
    for a in orders:
        eps = steps * rdp_sampled_gaussian(q, sigma, a) + math.log(1.0 / delta) / (a - 1)
        best = min(best, eps)
    return best


# ----------------------------------------------------------------------------
# Local training
# ----------------------------------------------------------------------------
def local_train(W0, Z, y, n_classes, epochs, batch, lr, rng, sigma=0.0, clip=None):
    """
    Mini-batch SGD on a softmax model.
    If clip is None and sigma == 0 -> ordinary SGD (uniform sampling).
    Otherwise DP-SGD with Poisson sampling, per-example clipping to `clip`
    and Gaussian noise of std sigma*clip on the summed gradient.
    Returns (W, n_steps, q).
    """
    W = W0.copy()
    n = len(y)
    Y = np.eye(n_classes)[y]
    steps = epochs * int(math.ceil(n / batch))
    q = min(1.0, batch / n)
    dp = clip is not None
    for _ in range(steps):
        if dp:
            idx = np.where(rng.rand(n) < q)[0]
            if len(idx) > 0:
                Zb = Z[idx]
                R = softmax(Zb @ W) - Y[idx]
                norms = np.linalg.norm(Zb, axis=1) * np.linalg.norm(R, axis=1)
                s = np.minimum(1.0, clip / np.maximum(norms, 1e-12))
                G = Zb.T @ (R * s[:, None])
            else:
                G = np.zeros_like(W)
            if sigma > 0:
                G = G + rng.normal(0.0, sigma * clip, size=G.shape)
            W -= lr * G / batch
        else:
            idx = rng.choice(n, size=min(batch, n), replace=False)
            Zb = Z[idx]
            R = softmax(Zb @ W) - Y[idx]
            W -= lr * (Zb.T @ R) / len(idx)
    return W, steps, q


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def fedavg(models, sizes):
    sizes = np.asarray(sizes, dtype=float)
    w = sizes / sizes.sum()
    return sum(wi * m for wi, m in zip(w, models))


def median_agg(models):
    return np.median(np.stack(models, axis=0), axis=0)


def secure_aggregate(updates, rng, mask_scale=10.0):
    """
    Additive-mask secure aggregation simulation (Bonawitz-style pairwise masks).
    Client i adds +m_ij for j>i and -m_ji for j<i; masks cancel in the sum.
    Returns (aggregate_sum, max_abs_error_vs_plain_sum, elapsed_seconds).
    """
    t0 = time.perf_counter()
    K = len(updates)
    flat = [u.ravel() for u in updates]
    masked = [f.copy() for f in flat]
    for i in range(K):
        for j in range(i + 1, K):
            m = rng.normal(0.0, mask_scale, size=flat[0].shape)
            masked[i] += m
            masked[j] -= m
    agg = np.sum(masked, axis=0)
    elapsed = time.perf_counter() - t0
    err = float(np.max(np.abs(agg - np.sum(flat, axis=0))))
    return agg.reshape(updates[0].shape), err, elapsed


# ----------------------------------------------------------------------------
# Membership inference attack (loss threshold)
# ----------------------------------------------------------------------------
def mia_auc(W, Z_train, y_train, Z_test, y_test, rng, max_n=1000):
    n = min(len(y_train), len(y_test), max_n)
    it = rng.choice(len(y_train), n, replace=False)
    ie = rng.choice(len(y_test), n, replace=False)
    l_in = per_example_loss(W, Z_train[it], y_train[it])
    l_out = per_example_loss(W, Z_test[ie], y_test[ie])
    scores = -np.concatenate([l_in, l_out])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return float(roc_auc_score(labels, scores))


# ----------------------------------------------------------------------------
# Federated training loop
# ----------------------------------------------------------------------------
def run_federated(Z_tr, y_tr, Z_te, y_te, parts_tr, parts_te, n_classes,
                  rounds=30, epochs=2, batch=32, lr=2.0, sigma=0.0, clip=None,
                  aggregator="fedavg", malicious=(), secure=False, seed=0,
                  record_curve=False):
    rng = np.random.RandomState(seed)
    d = Z_tr.shape[1]
    W = np.zeros((d, n_classes))
    K = len(parts_tr)
    sizes = [len(p) for p in parts_tr]
    curve, round_times, comm_bytes = [], [], 0
    max_q, total_steps = 0.0, 0
    sec_err, sec_time = 0.0, 0.0
    for r in range(rounds):
        t0 = time.perf_counter()
        models = []
        for k in range(K):
            idx = parts_tr[k]
            yk = y_tr[idx]
            if k in malicious:
                yk = (yk + 1) % n_classes  # label-flip attack
            Wk, steps, q = local_train(W, Z_tr[idx], yk, n_classes, epochs, batch, lr, rng,
                                       sigma=sigma, clip=clip)
            models.append(Wk)
            max_q = max(max_q, q)
        total_steps = steps  # steps per round for worst client counted below
        if secure:
            deltas = [m - W for m in models]
            w = np.asarray(sizes, float) / np.sum(sizes)
            weighted = [wi * dlt for wi, dlt in zip(w, deltas)]
            agg, err, et = secure_aggregate(weighted, rng)
            sec_err = max(sec_err, err)
            sec_time += et
            W = W + agg
        elif aggregator == "median":
            W = median_agg(models)
        else:
            W = fedavg(models, sizes)
        comm_bytes += 2 * K * W.size * 8  # download + upload, float64
        round_times.append(time.perf_counter() - t0)
        if record_curve:
            curve.append(accuracy(W, Z_te, y_te))
    # per-client test accuracy
    client_acc = []
    for k in range(K):
        ti = parts_te[k]
        if len(ti) > 0:
            client_acc.append(accuracy(W, Z_te[ti], y_te[ti]))
    out = dict(W=W, acc=accuracy(W, Z_te, y_te),
               f1=float(f1_score(y_te, predict(W, Z_te), average="macro")),
               client_acc=client_acc, round_time=float(np.mean(round_times)),
               comm_bytes=int(comm_bytes), curve=curve, q=max_q,
               sec_err=sec_err, sec_time=sec_time)
    return out


def steps_per_client(parts_tr, rounds, epochs, batch):
    """Return (max_q, total_steps) for worst-case client accounting."""
    n_min = min(len(p) for p in parts_tr)
    q = min(1.0, batch / n_min)
    steps = rounds * epochs * int(math.ceil(n_min / batch))
    return q, steps

"""
run_experiments.py - reproduces every number reported in Sections 13-17 of the report.
Usage:  python run_experiments.py   (writes ../03_Datasets/results.json and ../03_Datasets/results_summary_E1.csv)
All datasets are bundled with scikit-learn (no download needed).
"""
import json, os, time, math, platform, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fl_core import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03_Datasets")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
R, E, B = 20, 1, 16          # rounds, local epochs, batch size
LR_NODP, LR_DP, CLIP, DELTA = 10.0, 5.0, 1.0, 1e-5
SIGMAS = [0.8, 1.0, 1.5, 2.0, 3.0]


def load(name, seed):
    if name == "digits":
        d = load_digits(); X, y = d.data / 16.0, d.target
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        K, alpha, D = 10, 0.5, 256
    else:
        d = load_breast_cancer(); X, y = d.data, d.target
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        K, alpha, D = 5, 1.0, 128
    rff = RFFMap(Xtr.shape[1], D, gamma=2.0 / Xtr.shape[1], seed=seed)
    return Xtr, Xte, ytr, yte, rff(Xtr), rff(Xte), K, alpha, len(np.unique(y)), rff


def worst_eps(parts, sigma):
    eps = 0.0
    for p in parts:
        q = min(1.0, B / len(p)); steps = R * E * int(math.ceil(len(p) / B))
        eps = max(eps, epsilon_from_rdp(q, sigma, steps, DELTA))
    return eps


def mean_std(v):
    v = np.asarray(v, float); return [float(v.mean()), float(v.std())]


results = {"meta": dict(python=sys.version.split()[0], platform=platform.platform(),
                        numpy=np.__version__, rounds=R, local_epochs=E, batch=B,
                        lr_nodp=LR_NODP, lr_dp=LR_DP, clip=CLIP, delta=DELTA, seeds=SEEDS)}

# ---------------------------------------------------------------- E1 utility/privacy
e1 = {}
for ds in ["digits", "cancer"]:
    methods = {}
    def add(m, **kw):
        methods.setdefault(m, {k: [] for k in ["acc", "f1", "eps", "mia", "min_client", "comm", "round_time", "sec_err", "sec_time"]})
        for k, v in kw.items(): methods[m][k].append(v)
    for s in SEEDS:
        Xtr, Xte, ytr, yte, Ztr, Zte, K, alpha, C, rff = load(ds, s)
        ptr, pte = dirichlet_partition(ytr, yte, K, alpha, s)
        rng = np.random.RandomState(100 + s)
        # centralised cloud baseline
        Wc, _, _ = local_train(np.zeros((Ztr.shape[1], C)), Ztr, ytr, C, R * E, B, LR_NODP, rng)
        add("Centralised cloud", acc=accuracy(Wc, Zte, yte), f1=float(f1_score(yte, predict(Wc, Zte), average="macro")),
            eps=float("inf"), mia=mia_auc(Wc, Ztr, ytr, Zte, yte, rng), min_client=float(min(accuracy(Wc, Zte[t], yte[t]) for t in pte if len(t))),
            comm=int(Xtr.size * 8), round_time=float("nan"), sec_err=0.0, sec_time=0.0)
        # local-only edge
        accs, mia, mc = [], [], []
        for k in range(K):
            Wk, _, _ = local_train(np.zeros((Ztr.shape[1], C)), Ztr[ptr[k]], ytr[ptr[k]], C, R * E, B, LR_NODP, rng)
            accs.append(accuracy(Wk, Zte, yte)); mc.append(accuracy(Wk, Zte[pte[k]], yte[pte[k]]) if len(pte[k]) else np.nan)
        add("Local-only edge", acc=float(np.mean(accs)), f1=float("nan"), eps=float("inf"), mia=float("nan"),
            min_client=float(np.nanmin(mc)), comm=0, round_time=float("nan"), sec_err=0.0, sec_time=0.0)
        # FedAvg
        r = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_NODP, seed=s)
        add("FedAvg", acc=r["acc"], f1=r["f1"], eps=float("inf"), mia=mia_auc(r["W"], Ztr, ytr, Zte, yte, rng),
            min_client=min(r["client_acc"]), comm=r["comm_bytes"], round_time=r["round_time"], sec_err=0.0, sec_time=0.0)
        # FedAvg + secure aggregation
        r = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_NODP, secure=True, seed=s)
        add("FedAvg + SecAgg", acc=r["acc"], f1=r["f1"], eps=float("inf"), mia=mia_auc(r["W"], Ztr, ytr, Zte, yte, rng),
            min_client=min(r["client_acc"]), comm=r["comm_bytes"], round_time=r["round_time"], sec_err=r["sec_err"], sec_time=r["sec_time"])
        # DP-FedAvg (+SecAgg) sweeps
        for sg in SIGMAS:
            r = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_DP, sigma=sg, clip=CLIP, secure=True, seed=s)
            add(f"DP sigma={sg}", acc=r["acc"], f1=r["f1"], eps=worst_eps(ptr, sg),
                mia=mia_auc(r["W"], Ztr, ytr, Zte, yte, rng), min_client=min(r["client_acc"]),
                comm=r["comm_bytes"], round_time=r["round_time"], sec_err=r["sec_err"], sec_time=r["sec_time"])
    e1[ds] = {m: {k: (mean_std(v) if k not in ("eps",) else float(np.mean(v))) for k, v in d.items()} for m, d in methods.items()}
    print(ds, {m: round(v["acc"][0], 3) for m, v in e1[ds].items()}, flush=True)
results["E1_utility_privacy"] = e1

# convergence curves (digits, seed 0)
Xtr, Xte, ytr, yte, Ztr, Zte, K, alpha, C, rff = load("digits", 0)
ptr, pte = dirichlet_partition(ytr, yte, K, alpha, 0)
curves = {}
curves["FedAvg"] = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_NODP, seed=0, record_curve=True)["curve"]
for sg in [0.8, 1.5, 3.0]:
    curves[f"DP sigma={sg}"] = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_DP, sigma=sg, clip=CLIP, seed=0, record_curve=True)["curve"]
results["E1_curves"] = curves

# ---------------------------------------------------------------- MIA probe (over-parameterised, small train set)
mia = {"settings": [], "auc": [], "acc": []}
for label, sg in [("No DP", None)] + [(f"sigma={s}", s) for s in SIGMAS]:
    aucs, accs = [], []
    for s in SEEDS:
        d = load_digits(); X, y = d.data / 16.0, d.target
        Xtr_, Xte_, ytr_, yte_ = train_test_split(X, y, test_size=0.7, random_state=s, stratify=y)
        rf = RFFMap(64, 1024, gamma=2.0 / 64, seed=s); Za, Zb = rf(Xtr_), rf(Xte_)
        p1, p2 = dirichlet_partition(ytr_, yte_, 10, 0.5, s)
        if sg is None:
            r = run_federated(Za, ytr_, Zb, yte_, p1, p2, 10, 40, 3, B, 20.0, seed=s)
        else:
            r = run_federated(Za, ytr_, Zb, yte_, p1, p2, 10, 40, 3, B, 5.0, sigma=sg, clip=CLIP, seed=s)
        aucs.append(mia_auc(r["W"], Za, ytr_, Zb, yte_, np.random.RandomState(7 + s))); accs.append(r["acc"])
    mia["settings"].append(label); mia["auc"].append(mean_std(aucs)); mia["acc"].append(mean_std(accs))
results["E1b_mia_probe"] = mia
print("MIA probe", mia["auc"], flush=True)

# ---------------------------------------------------------------- E2 non-IID
e2 = {"alpha": [0.1, 0.3, 0.5, 1.0, 5.0, 100.0], "acc": [], "min_client": []}
for a in e2["alpha"]:
    accs, mins = [], []
    for s in SEEDS:
        Xtr, Xte, ytr, yte, Ztr, Zte, K, _, C, _ = load("digits", s)
        ptr, pte = dirichlet_partition(ytr, yte, K, a, s)
        r = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_NODP, seed=s)
        accs.append(r["acc"]); mins.append(min(r["client_acc"]))
    e2["acc"].append(mean_std(accs)); e2["min_client"].append(mean_std(mins))
results["E2_noniid"] = e2

# ---------------------------------------------------------------- E3 scalability
e3 = {"K": [5, 10, 20, 40, 80], "round_plain": [], "round_secagg": [], "secagg_overhead": [], "server_bytes": []}
for Kc in e3["K"]:
    tp, ts, ov = [], [], []
    for s in SEEDS:
        d = load_digits(); X, y = d.data / 16.0, d.target
        Xtr_, Xte_, ytr_, yte_ = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
        rf = RFFMap(64, 256, gamma=2.0 / 64, seed=s); Za, Zb = rf(Xtr_), rf(Xte_)
        p1, p2 = dirichlet_partition(ytr_, yte_, Kc, 5.0, s)
        a = run_federated(Za, ytr_, Zb, yte_, p1, p2, 10, 5, E, B, LR_NODP, seed=s)
        b = run_federated(Za, ytr_, Zb, yte_, p1, p2, 10, 5, E, B, LR_NODP, secure=True, seed=s)
        tp.append(a["round_time"]); ts.append(b["round_time"]); ov.append(b["sec_time"] / 5)
    e3["round_plain"].append(mean_std(tp)); e3["round_secagg"].append(mean_std(ts)); e3["secagg_overhead"].append(mean_std(ov))
    e3["server_bytes"].append(int(2 * Kc * (Za.shape[1] * 10) * 8))
results["E3_scalability"] = e3

# ---------------------------------------------------------------- E4 robustness
e4 = {"frac": [0.0, 0.1, 0.2, 0.3, 0.4], "FedAvg": [], "Median": [], "DP sigma=1.0": []}
for f in e4["frac"]:
    res = {"FedAvg": [], "Median": [], "DP sigma=1.0": []}
    for s in SEEDS:
        Xtr, Xte, ytr, yte, Ztr, Zte, K, alpha, C, _ = load("digits", s)
        ptr, pte = dirichlet_partition(ytr, yte, K, 1.0, s)
        mal = tuple(range(int(round(f * K))))
        res["FedAvg"].append(run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_NODP, malicious=mal, seed=s)["acc"])
        res["Median"].append(run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_NODP, aggregator="median", malicious=mal, seed=s)["acc"])
        res["DP sigma=1.0"].append(run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_DP, sigma=1.0, clip=CLIP, malicious=mal, seed=s)["acc"])
    for k in res: e4[k].append(mean_std(res[k]))
results["E4_robustness"] = e4

# ---------------------------------------------------------------- E5 latency / offload (measured compute + modelled network)
Xtr, Xte, ytr, yte, Ztr, Zte, K, alpha, C, rff = load("digits", 0)
ptr, pte = dirichlet_partition(ytr, yte, K, alpha, 0)
edge = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_NODP, seed=0)
rff_cloud = RFFMap(64, 1024, gamma=2.0 / 64, seed=0)
Zc_tr, Zc_te = rff_cloud(Xtr), rff_cloud(Xte)
Wcloud, _, _ = local_train(np.zeros((Zc_tr.shape[1], C)), Zc_tr, ytr, C, 40, B, LR_NODP, np.random.RandomState(5))
# measured single-sample inference latency on this CPU
reps = 3000
xs = Xte[np.random.RandomState(0).randint(0, len(Xte), reps)]
t0 = time.perf_counter()
for i in range(reps): predict(edge["W"], rff(xs[i:i + 1]))
t_edge_meas = (time.perf_counter() - t0) / reps
t0 = time.perf_counter()
for i in range(reps): predict(Wcloud, rff_cloud(xs[i:i + 1]))
t_cloud_meas = (time.perf_counter() - t0) / reps
conf = softmax(Zte @ edge["W"]).max(axis=1)
off = {}
for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    m = conf < tau
    pred = predict(edge["W"], Zte); pc = predict(Wcloud, Zc_te)
    hyb = np.where(m, pc, pred)
    off[str(tau)] = dict(offload=float(m.mean()), acc=float((hyb == yte).mean()))
results["E5_latency"] = dict(t_edge_meas_ms=t_edge_meas * 1e3, t_cloud_meas_ms=t_cloud_meas * 1e3,
                             edge_acc=float((predict(edge["W"], Zte) == yte).mean()),
                             cloud_acc=float((predict(Wcloud, Zc_te) == yte).mean()), offload=off,
                             model_bytes_edge=int(edge["W"].size * 8), model_bytes_cloud=int(Wcloud.size * 8))
print("latency", results["E5_latency"]["t_edge_meas_ms"], results["E5_latency"]["t_cloud_meas_ms"], flush=True)

# ---------------------------------------------------------------- E6 explainability (permutation importance on pixels)
def perm_importance(W, rf, X, y, seed=0):
    rng = np.random.RandomState(seed)
    base = accuracy(W, rf(X), y); imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp[j] = base - accuracy(W, rf(Xp), y)
    return imp
dp = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_DP, sigma=1.0, clip=CLIP, seed=0)
dp2 = run_federated(Ztr, ytr, Zte, yte, ptr, pte, C, R, E, B, LR_DP, sigma=2.0, clip=CLIP, seed=0)
Wc0, _, _ = local_train(np.zeros((Ztr.shape[1], C)), Ztr, ytr, C, R * E, B, LR_NODP, np.random.RandomState(9))
imp = {"Centralised": perm_importance(Wc0, rff, Xte, yte), "FedAvg": perm_importance(edge["W"], rff, Xte, yte),
       "DP sigma=1.0": perm_importance(dp["W"], rff, Xte, yte), "DP sigma=2.0": perm_importance(dp2["W"], rff, Xte, yte)}
results["E6_explain"] = dict(imp={k: v.tolist() for k, v in imp.items()},
                             rho_fedavg_central=float(spearmanr(imp["FedAvg"], imp["Centralised"])[0]),
                             rho_dp1_central=float(spearmanr(imp["DP sigma=1.0"], imp["Centralised"])[0]),
                             rho_dp2_central=float(spearmanr(imp["DP sigma=2.0"], imp["Centralised"])[0]))
print("explain rho", results["E6_explain"]["rho_fedavg_central"], results["E6_explain"]["rho_dp1_central"], flush=True)

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=1)
# flat summary of experiment E1 (one row per dataset and system)
with open(os.path.join(OUT, "results_summary_E1.csv"), "w") as f:
    f.write("dataset,system,acc_mean,acc_std,f1_mean,eps_worst_client,mia_auc,min_client_acc,comm_bytes,round_time_s\n")
    for ds, ms in e1.items():
        for m, v in ms.items():
            f.write(",".join(str(x) for x in [ds, m, v["acc"][0], v["acc"][1], v["f1"][0], v["eps"], v["mia"][0],
                                              v["min_client"][0], v["comm"][0], v["round_time"][0]]) + "\n")
print("saved", flush=True)

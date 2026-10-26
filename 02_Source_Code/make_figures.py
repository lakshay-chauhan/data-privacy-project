"""
make_figures.py - generates every figure at 600 dpi from results/results.json.
Diagrams are produced with Graphviz (DOT sources are also saved so they can be re-drawn in draw.io).
"""
import json, os, subprocess, math, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = json.load(open(os.path.join(BASE, "results", "results.json")))
EXTRA = json.load(open(os.path.join(BASE, "results", "extra_robustness.json")))
FIG = os.path.join(BASE, "diagrams"); os.makedirs(FIG, exist_ok=True)
DOT = os.path.join(BASE, "diagrams", "dot_sources"); os.makedirs(DOT, exist_ok=True)
FONT = "Arial"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
                     "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "savefig.dpi": 600, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4})
C = dict(edge="#2F6DB5", cloud="#E07B39", gov="#3A9D6E", sec="#8E5BB5", gray="#6B7280", red="#C0392B", dark="#1F2937")

NAMES = {}  # fig number -> (title, filename)


def fname(n, title):
    safe = title.replace("/", "-").replace(":", "").replace("&", "and")
    return f"Fig {n:02d} {safe}.png"


# ---------------------------------------------------------------- Graphviz helpers
HEAD = ('digraph G {{ graph [fontname="{f}", fontsize=9, bgcolor="white", pad=0.04, nodesep=0.14, ranksep=0.2, '
        'compound=true, size="3.45,8.2", splines=true]; node [shape=box, style="rounded,filled", fontname="{f}", '
        'fontsize=8.5, fillcolor="#EAF1FB", color="#2F6DB5", penwidth=0.8, margin="0.07,0.035"]; '
        'edge [color="#4B5563", arrowsize=0.6, penwidth=0.8, fontname="{f}", fontsize=7.5]; rankdir={rd};\n')
LAYER_FILL = ["#EAF1FB", "#FDF0E6", "#E9F6EF", "#F2EAF8", "#F3F4F6", "#FFF8DB", "#E6F4F7"]
LAYER_LINE = ["#2F6DB5", "#E07B39", "#3A9D6E", "#8E5BB5", "#6B7280", "#B8860B", "#2A8CA0"]


def _find_dot():
    """Find dot executable: check PATH first, then common Windows install dirs."""
    on_path = shutil.which("dot")
    if on_path:
        return on_path
    candidates = [
        # Portable extraction bundled with this project
        os.path.join(BASE, "graphviz_portable", "Graphviz-16.1.0-win64", "bin", "dot.exe"),
        r"C:\Program Files\Graphviz\bin\dot.exe",
        r"C:\Program Files (x86)\Graphviz\bin\dot.exe",
        r"C:\Graphviz\bin\dot.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Graphviz\bin\dot.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Graphviz 'dot' not found. Install from https://graphviz.org/download/ "
        "or run: winget install Graphviz.Graphviz"
    )

DOT_EXE = _find_dot()


def render(n, title, dot):
    out = os.path.join(FIG, fname(n, title))
    src = os.path.join(DOT, fname(n, title).replace(".png", ".dot"))
    open(src, "w").write(dot)
    subprocess.run([DOT_EXE, "-Tpng", "-Gdpi=600", src, "-o", out], check=True)
    NAMES[n] = (title, out)


def q(s):
    return s.replace('"', "'")


def chain(n, title, steps, loop=None, rd="TB", labels=None):
    d = HEAD.format(f=FONT, rd=rd)
    for i, s in enumerate(steps):
        c = i % len(LAYER_FILL)
        d += f'n{i} [label="{q(s)}", fillcolor="{LAYER_FILL[c]}", color="{LAYER_LINE[c]}"];\n'
    for i in range(len(steps) - 1):
        lab = f' [label="{q(labels[i])}"]' if labels and i < len(labels) and labels[i] else ""
        d += f"n{i} -> n{i+1}{lab};\n"
    if loop:
        d += f'n{len(steps)-1} -> n0 [style=dashed, constraint=false, label="{q(loop)}", color="#E07B39", fontcolor="#E07B39"];\n'
    render(n, title, d + "}")


def stack(n, title, layers, cross=None, arrows="down"):
    """layers: list of (title, [items]); drawn top to bottom, each layer a cluster."""
    d = HEAD.format(f=FONT, rd="TB")
    first = []
    for li, (lt, items) in enumerate(layers):
        c = li % len(LAYER_FILL)
        d += (f'subgraph cluster_{li} {{ label="{q(lt)}"; labeljust=l; fontsize=8.5; style="rounded,filled"; '
              f'fillcolor="{LAYER_FILL[c]}"; color="{LAYER_LINE[c]}"; penwidth=0.9; margin=7;\n')
        d += "{rank=same; "
        for ii, it in enumerate(items):
            d += f'l{li}i{ii} [label="{q(it)}", fillcolor="white", color="{LAYER_LINE[c]}"]; '
        d += "}\n"
        for ii in range(len(items) - 1):
            d += f"l{li}i{ii} -> l{li}i{ii+1} [style=invis];\n"
        d += "}\n"
        first.append((li, len(items) // 2))
    for li in range(len(layers) - 1):
        a, b = first[li], first[li + 1]
        sty = {"down": "", "up": ", dir=back", "none": ", style=invis"}[arrows]
        d += (f"l{a[0]}i{a[1]} -> l{b[0]}i{b[1]} [ltail=cluster_{a[0]}, lhead=cluster_{b[0]}, "
              f'color="#4B5563", penwidth=1.1{sty}];\n')
    if cross:
        for a, b, lab in cross:
            d += f'l{a}i0 -> l{b}i0 [ltail=cluster_{a}, lhead=cluster_{b}, style=dashed, constraint=false, label="{q(lab)}", color="#E07B39", fontcolor="#E07B39"];\n'
    render(n, title, d + "}")


def tree(n, title, root, groups):
    d = HEAD.format(f=FONT, rd="LR").replace("nodesep=0.14", "nodesep=0.035").replace("ranksep=0.2", "ranksep=0.25")
    d += f'root [label="{q(root)}", fillcolor="#1F2937", fontcolor="white", color="#1F2937"];\n'
    for gi, (g, leaves) in enumerate(groups):
        c = gi % len(LAYER_FILL)
        d += f'g{gi} [label="{q(g)}", fillcolor="{LAYER_FILL[c]}", color="{LAYER_LINE[c]}", margin="0.06,0.025"];\nroot -> g{gi} [arrowsize=0.4];\n'
        for li, lf in enumerate(leaves):
            d += f'g{gi}l{li} [label="{q(lf)}", fillcolor="white", color="{LAYER_LINE[c]}", fontsize=7.5, margin="0.05,0.015", height=0.25];\ng{gi} -> g{gi}l{li} [arrowsize=0.4];\n'
    render(n, title, d + "}")


def custom(n, title, body, rd="TB"):
    render(n, title, HEAD.format(f=FONT, rd=rd) + body + "}")


# ================================================================ DIAGRAMS
chain(1, "Evolution of Cloud-Edge Analytics Architectures", [
    "Centralised cloud analytics\nraw data sent to data centres",
    "Cloud + fog / CDN offload\nregional caching and pre-processing",
    "Edge computing\nlocal inference near the data",
    "Edge intelligence\non-device AI and model compression",
    "Privacy-aware cloud-edge intelligence\nfederated analytics + DP + governance"])

stack(2, "Distributed Data Science Ecosystem Overview", [
    ("Data sources", ["IoT sensors", "Mobile devices", "Enterprise systems"]),
    ("Edge tier", ["Gateways", "Edge AI nodes"]),
    ("Cloud tier", ["Data engineering", "Analytics and AI", "Model registry"]),
    ("Consumers", ["Dashboards", "Decision services", "Regulators"]),
    ("Cross-cutting controls", ["Privacy and security", "Governance and audit"])])

tree(3, "Privacy Challenges in Cloud-Edge Data Science Systems", "Privacy\nchallenges", [
    ("Data leakage", ["Membership inference", "Gradient inversion"]),
    ("Central dependency", ["Raw data transfer", "Single point of failure"]),
    ("Trust deficit", ["Opaque models", "Unverifiable updates"]),
    ("Resource limits", ["Edge compute and energy", "Latency and bandwidth"]),
    ("Governance", ["GDPR, DPDP Act, AI Act", "Audit and erasure duties"])])

tree(4, "Literature Taxonomy Framework", "Privacy-aware\ncloud-edge\nintelligence", [
    ("Edge intelligence", ["Edge AI surveys", "LLMs at the edge", "TinyML and MLOps"]),
    ("Federated learning", ["Edge and IoT FL", "Split learning", "Trustworthy FL"]),
    ("Privacy techniques", ["Differential privacy", "Secure aggregation / HE", "TEEs, anonymisation"]),
    ("Trust and governance", ["XAI and trust models", "Unlearning, audit", "Requirements, law"]),
    ("Applications", ["Health, smart city", "Industry, energy", "Transport, agriculture"])])

chain(5, "Requirements Engineering Framework", [
    "Elicit stakeholder, legal\nand operational needs",
    "Classify: privacy, security,\nperformance, trust, governance",
    "Specify measurable\nrequirements and thresholds",
    "Map to architecture layers\nand privacy controls",
    "Verify with metrics,\ntests and audits"], loop="refine")

stack(6, "Privacy-Aware Cloud-Edge Intelligence Architecture for Distributed Data Science", [
    ("Decision and application layer", ["Decision support", "Dashboards and APIs"]),
    ("Governance and policy plane", ["Policy engine", "Audit log", "Consent / erasure"]),
    ("Cloud intelligence", ["Federated aggregator", "Data lakehouse", "Analytics and AI", "Trust engine"]),
    ("Secure communication", ["mTLS channel", "Secure aggregation"]),
    ("Edge intelligence", ["Local training\n(clip + noise)", "Edge inference", "Anonymisation"]),
    ("Data acquisition", ["Sensors", "Gateways", "Local stores"])], arrows="up")

chain(7, "Multi-Layer Architectural View", [
    "L1  Data acquisition and edge\nintelligence (Section 6)",
    "L2  Distributed data engineering\n(Section 7)",
    "L3  Privacy preservation and\nsecurity (Section 8)",
    "L4  Federated analytics and\nlearning (Section 9)",
    "L5  Cloud intelligence and\nanalytics (Section 10)",
    "L6  Explainability, trust and\ndecisions (Section 11)",
    "L7  Governance, compliance\nand policy (Section 12)"], rd="BT")

chain(8, "Edge Data Acquisition Framework", [
    "Sensors and IoT devices",
    "Gateway: protocol translation\n(MQTT, CoAP, HTTP)",
    "Quality checks and\nlocal filtering",
    "Minimisation and\npseudonymisation",
    "Local buffer / edge store"])

custom(9, "Edge Intelligence Processing Pipeline", """
a [label="Raw signal"]; b [label="Pre-processing and\\nfeature extraction", fillcolor="#FDF0E6", color="#E07B39"];
c [label="Local inference\\n(compressed model)", fillcolor="#E9F6EF", color="#3A9D6E"];
d [label="Confidence\\n>= threshold?", shape=diamond, style="filled", fillcolor="#FFF8DB", color="#B8860B", margin="0.02,0.02"];
e [label="Act locally\\n(no data leaves)", fillcolor="#E9F6EF", color="#3A9D6E"];
f [label="Send features only\\nto cloud model", fillcolor="#F2EAF8", color="#8E5BB5"];
g [label="Cloud inference\\nand response", fillcolor="#F2EAF8", color="#8E5BB5"];
a -> b -> c -> d; d -> e [label="yes"]; d -> f [label="no"]; f -> g;
""")

chain(10, "Distributed Data Engineering Workflow", [
    "Ingest (stream and batch)",
    "Validate and deduplicate",
    "Transform and feature store",
    "Metadata and lineage catalogue",
    "Edge-cloud synchronisation\n(delta updates)",
    "Serve to analytics"])

custom(11, "Edge-to-Cloud Data Flow Architecture", """
a [label="Devices"]; b [label="Edge node\\nlocal store + model", fillcolor="#FDF0E6", color="#E07B39"];
c [label="Secure channel\\nfeatures / masked updates only", fillcolor="#F2EAF8", color="#8E5BB5"];
d [label="Cloud ingestion", fillcolor="#E9F6EF", color="#3A9D6E"]; e [label="Lakehouse + feature store", fillcolor="#E9F6EF", color="#3A9D6E"];
f [label="Analytics and\\nfederated aggregation", fillcolor="#E9F6EF", color="#3A9D6E"];
a -> b -> c -> d -> e -> f; f -> b [style=dashed, constraint=false, label="global model", color="#E07B39", fontcolor="#E07B39"];
""")

tree(12, "Privacy Preservation Framework", "Privacy\npreservation", [
    ("Data level", ["Minimisation", "Anonymisation"]),
    ("Computation level", ["Differential privacy", "Secure aggregation, HE", "Trusted execution (TEE)"]),
    ("Access level", ["RBAC / ABAC", "Consent management"]),
    ("Communication level", ["mTLS", "Authenticated devices"]),
    ("Lifecycle level", ["Retention limits", "Unlearning / erasure"])])

chain(13, "Secure Communication and Storage Architecture", [
    "Edge node: device identity\nand local key store",
    "mTLS tunnel\n(encrypted, authenticated)",
    "API gateway and\naccess control",
    "Encrypted storage\n(per-tenant keys)",
    "Key management service\nand audit log"])

stack(14, "Federated Analytics Architecture", [
    ("Edge clients", ["Local data", "Local training", "Clip + noise"]),
    ("Secure aggregation", ["Pairwise masks", "Masked updates"]),
    ("Aggregator", ["FedAvg / median", "Privacy accountant"]),
    ("Model registry and monitor", ["Global model", "Trust monitor"])],
    cross=[(3, 0, "broadcast next round")])

chain(15, "Federated Learning Workflow", [
    "1  Server broadcasts the\nglobal model to clients",
    "2  Clients train on\nprivate local data",
    "3  Clip update and add\nGaussian noise (DP)",
    "4  Mask update\n(secure aggregation)",
    "5  Server sums updates;\nmasks cancel; FedAvg",
    "6  Evaluate, log and\nstart next round"], loop="next round")

stack(16, "Cloud Analytics Framework", [
    ("Ingestion", ["Streaming", "Batch"]),
    ("Storage", ["Lakehouse", "Feature store"]),
    ("Processing", ["Stream analytics", "Batch analytics"]),
    ("Model lifecycle", ["Training", "Registry"]),
    ("Serving", ["Inference API", "Monitoring"])])

chain(17, "AI-Powered Intelligence Pipeline", [
    "Features and labels",
    "Training (federated or central)",
    "Validation: accuracy,\nprivacy, fairness",
    "Compression and packaging",
    "Staged deployment to edge",
    "Monitoring and drift detection"], loop="retrain")

stack(18, "Explainability and Trust Framework", [
    ("Model outputs", ["Predictions", "Confidence"]),
    ("Explainers", ["Feature importance", "Example-based"]),
    ("Trust assessment", ["Utility", "Privacy", "Fairness", "Robustness"]),
    ("Transparency", ["Model card", "Audit report"]),
    ("Human oversight", ["Review queue", "Override"])])

chain(19, "Decision Intelligence Architecture", [
    "Predictions + explanations\n+ trust score",
    "Trust-aware policy rules",
    "Recommendation engine",
    "Decision-support dashboard",
    "Action and outcome feedback"], loop="feedback")

stack(20, "Governance and Compliance Architecture", [
    ("Regulation", ["GDPR", "DPDP Act 2023", "EU AI Act"]),
    ("Policy management", ["Policy-as-code", "Risk register"]),
    ("Enforcement points", ["Edge", "Gateway", "Cloud"]),
    ("Monitoring", ["Audit trail", "Compliance reports"]),
    ("Data-subject rights", ["Access", "Erasure / unlearning"])])

stack(21, "Experimental Testbed Architecture", [
    ("Datasets (scikit-learn, public)", ["Digits (image)", "Breast cancer (clinical)"]),
    ("Partitioner", ["Dirichlet non-IID", "5 or 10 clients"]),
    ("Emulated edge clients (NumPy)", ["RFF + softmax", "DP-SGD", "Label-flip attackers"]),
    ("Cloud aggregator", ["FedAvg / median", "SecAgg", "RDP accountant"]),
    ("Measurement", ["Accuracy, F1", "Epsilon, MIA AUC", "Time, bytes"])])

tree(22, "Evaluation Framework", "Evaluation", [
    ("Utility", ["Accuracy", "Macro F1"]),
    ("Privacy", ["Epsilon (delta 1e-5)", "MIA AUC", "Data exposure"]),
    ("Performance", ["Round time", "Latency, throughput", "Communication"]),
    ("Trust", ["Fairness", "Robustness", "Trust score"]),
    ("Governance", ["Policy coverage", "Audit completeness"])])

stack(27, "Security and Reliability Framework", [
    ("Threats", ["Poisoning", "Inference attacks", "Node failure"]),
    ("Defences", ["Robust aggregation", "DP + SecAgg", "TLS, replication"]),
    ("Resilience", ["Local fallback", "Graceful degradation"]),
    ("Assurance", ["Metrics", "Audits", "Incident response"])])

stack(28, "Smart City Cloud-Edge Ecosystem", [
    ("City sensors", ["Traffic cameras", "Air quality", "Smart meters"]),
    ("Street-level edge nodes", ["On-device anonymisation", "Local analytics"]),
    ("Regional edge cloud", ["Federated aggregation"]),
    ("City cloud analytics", ["Digital twin", "Forecasting"]),
    ("Services", ["Traffic", "Energy", "Public safety"])])

chain(29, "Healthcare Edge Intelligence Framework", [
    "Wearables and bedside devices",
    "Hospital edge: de-identification\nand local training",
    "Secure aggregation across\nhospitals (DP noise)",
    "Cloud research model\nand audit trail",
    "Clinical decision support\nwith explanations"])

chain(31, "Challenges and Limitations Roadmap", [
    "Edge resource limits\n-> compression, TinyML",
    "Communication overhead\n-> sparsification, fewer rounds",
    "Privacy-utility trade-off\n-> tuned noise, better accounting",
    "Non-IID data and fairness\n-> personalisation",
    "Deployment and governance\n-> policy-as-code, MLOps"])

chain(32, "Future Research Roadmap", [
    "Now: robust aggregation and\ncalibrated differential privacy",
    "Next: privacy-preserving edge\nfoundation models",
    "Then: agentic edge intelligence\nwith confidential computing",
    "Later: quantum-safe edge\nanalytics",
    "Beyond: cognitive distributed\nsystems"])

stack(33, "Next-Generation Cloud-Edge Intelligence Ecosystem", [
    ("Agentic edge assistants", ["On-device small models", "Tool-use with consent"]),
    ("Confidential computing", ["Enclaves", "Remote attestation"]),
    ("Federated foundation models", ["Parameter-efficient tuning", "Secure aggregation"]),
    ("Post-quantum security", ["PQC key exchange", "Crypto-agility"]),
    ("Continuous compliance", ["Policy-as-code", "Unlearning"])], arrows="none")

tree(34, "Research Contribution Framework", "Contributions", [
    ("C1 Architecture", ["Seven-layer design"]),
    ("C2 Privacy toolkit", ["DP + SecAgg + RDP accounting"]),
    ("C3 Testbed", ["Open code and results"]),
    ("C4 Trust model", ["Six-pillar trust score"]),
    ("C5 Governance", ["Requirement-to-control map"])])


# ================================================================ DATA FIGURES
def save(fig, n, title):
    out = os.path.join(FIG, fname(n, title))
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    NAMES[n] = (title, out)


E1 = RES["E1_utility_privacy"]
SIG = [0.8, 1.0, 1.5, 2.0, 3.0]
DER = {}

# ---- Fig 23 privacy performance (wide, 1x3)
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.45))
for ds, col, lab in (("digits", C["edge"], "Digits"), ("cancer", C["cloud"], "Breast cancer")):
    eps = [E1[ds][f"DP sigma={s}"]["eps"] for s in SIG]
    acc = [E1[ds][f"DP sigma={s}"]["acc"][0] for s in SIG]
    sd = [E1[ds][f"DP sigma={s}"]["acc"][1] for s in SIG]
    ax[0].errorbar(eps, acc, yerr=sd, marker="o", ms=3, lw=1, capsize=2, color=col, label=lab)
    ax[0].axhline(E1[ds]["FedAvg"]["acc"][0], color=col, ls="--", lw=0.7)
ax[0].set_xlabel(r"Privacy budget $\varepsilon$ ($\delta = 10^{-5}$, worst-case client)"); ax[0].set_ylabel("Test accuracy")
ax[0].set_title("(a) Accuracy versus \u03b5 (dashed: FedAvg, no DP)", pad=3); ax[0].legend(frameon=False)
for lab, col in (("FedAvg", C["dark"]), ("DP sigma=0.8", C["edge"]), ("DP sigma=1.5", C["cloud"]), ("DP sigma=3.0", C["red"])):
    ax[1].plot(range(1, 21), RES["E1_curves"][lab.replace("FedAvg", "FedAvg")], color=col, lw=1, label=lab.replace("sigma", "\u03c3"))
ax[1].set_xlabel("Communication round"); ax[1].set_ylabel("Test accuracy"); ax[1].set_title("(b) Convergence (digits)", pad=3)
ax[1].legend(frameon=False, loc="lower right")
mp = RES["E1b_mia_probe"]; x = np.arange(len(mp["settings"]))
ax[2].bar(x, [a[0] for a in mp["auc"]], yerr=[a[1] for a in mp["auc"]], color=C["sec"], width=0.6, capsize=2, error_kw=dict(lw=0.6))
ax[2].axhline(0.5, color=C["red"], ls="--", lw=0.8); ax[2].set_ylim(0.4, 0.62)
ax[2].set_xticks(x); ax[2].set_xticklabels([s.replace("sigma", "\u03c3") for s in mp["settings"]], rotation=35, ha="right")
ax[2].set_ylabel("MIA AUC (0.5 = chance)"); ax[2].set_title("(c) Membership inference probe", pad=3)
fig.tight_layout(); save(fig, 23, "Privacy Preservation Performance Analysis")

# ---- derived: latency / throughput emulation (assumptions explicit)
L = RES["E5_latency"]
RTT, PAYLOAD, FEAT = 0.050, 64e3, 6.4e3          # s, bytes, bytes (assumed)
T_EDGE, T_CLOUD = 0.040, 0.004                   # s heavy-model scenario (assumed)
P_OFF = L["offload"]["0.7"]["offload"]
bw = np.logspace(0, 2, 60) * 1e6                 # 1-100 Mbps
lat_cloud = RTT + PAYLOAD * 8 / bw + T_CLOUD
lat_edge = np.full_like(bw, T_EDGE)
lat_hyb = T_EDGE + P_OFF * (RTT + FEAT * 8 / bw + T_CLOUD)
thr_cloud = bw / (PAYLOAD * 8)
thr_edge = np.full_like(bw, 1 / T_EDGE)
thr_hyb = np.minimum(1 / T_EDGE, bw / (P_OFF * FEAT * 8))
DER["latency_at_10mbps_ms"] = dict(cloud=float(np.interp(10e6, bw, lat_cloud) * 1e3), edge=T_EDGE * 1e3,
                                   hybrid=float(np.interp(10e6, bw, lat_hyb) * 1e3))
DER["throughput_at_10mbps"] = dict(cloud=float(np.interp(10e6, bw, thr_cloud)), edge=1 / T_EDGE,
                                   hybrid=float(np.interp(10e6, bw, thr_hyb)))
DER["latency_at_1mbps_ms"] = dict(cloud=float(np.interp(1e6, bw, lat_cloud) * 1e3), hybrid=float(np.interp(1e6, bw, lat_hyb) * 1e3))
DER["offload_tau07"] = P_OFF

# ---- Fig 24 latency and throughput (wide, 2x2)
fig, ax = plt.subplots(2, 2, figsize=(7.1, 4.6))
a = ax[0, 0]
a.plot(bw / 1e6, lat_cloud * 1e3, color=C["cloud"], label="Cloud-only"); a.plot(bw / 1e6, lat_edge * 1e3, color=C["edge"], label="Edge-only")
a.plot(bw / 1e6, lat_hyb * 1e3, color=C["gov"], label=f"Hybrid (offload {P_OFF*100:.0f}%)")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Uplink bandwidth (Mbps)"); a.set_ylabel("Mean latency (ms)")
a.set_title("(a) Latency versus bandwidth (emulated)", pad=3); a.legend(frameon=False)
a = ax[0, 1]
a.plot(bw / 1e6, thr_cloud, color=C["cloud"], label="Cloud-only"); a.plot(bw / 1e6, thr_edge, color=C["edge"], ls="--", lw=1.6, label="Edge-only")
a.plot(bw / 1e6, thr_hyb, color=C["gov"], label="Hybrid")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Uplink bandwidth (Mbps)"); a.set_ylabel("Sustainable throughput (requests/s)")
a.set_title("(b) Throughput versus bandwidth (emulated)", pad=3); a.legend(frameon=False)
a = ax[1, 0]
offs = [0.0] + [L["offload"][k]["offload"] for k in ("0.5", "0.6", "0.7", "0.8", "0.9", "0.95")] + [1.0]
accs = [L["edge_acc"]] + [L["offload"][k]["acc"] for k in ("0.5", "0.6", "0.7", "0.8", "0.9", "0.95")] + [L["cloud_acc"]]
a.plot(np.array(offs) * 100, accs, marker="o", ms=3, color=C["sec"]); a.set_xlabel("Samples offloaded to cloud (%)"); a.set_ylabel("Test accuracy")
a.set_title("(c) Accuracy versus offload fraction (measured)", pad=3)
a = ax[1, 1]; e3 = RES["E3_scalability"]; K = e3["K"]
a.errorbar(K, [v[0] * 1e3 for v in e3["round_plain"]], yerr=[v[1] * 1e3 for v in e3["round_plain"]], marker="o", ms=3, color=C["edge"], label="FedAvg", capsize=2)
a.errorbar(K, [v[0] * 1e3 for v in e3["round_secagg"]], yerr=[v[1] * 1e3 for v in e3["round_secagg"]], marker="s", ms=3, color=C["red"], label="FedAvg + SecAgg", capsize=2)
a.set_yscale("log"); a.set_xlabel("Number of clients"); a.set_ylabel("Time per round (ms)"); a.set_title("(d) Round time versus clients (measured)", pad=3); a.legend(frameon=False)
fig.tight_layout(); save(fig, 24, "Latency and Throughput Evaluation")

# ---- trust score model
raw = E1["digits"]["Centralised cloud"]["comm"][0]
SYS = {}
fed = E1["digits"]
rob = RES["E4_robustness"]; i0, i2 = rob["frac"].index(0.0), rob["frac"].index(0.2)
def pillars(name, acc, minc, eps, minim, rob_ratio, comm):
    return dict(Utility=acc, Minimisation=minim, Formal_privacy=(1.0 if eps == 0 else (0.0 if math.isinf(eps) else math.exp(-eps / 20.0))),
                Fairness=minc, Robustness=rob_ratio, Comm_efficiency=min(1.0, raw / comm) if comm > 0 else 1.0)
SYS["Centralised cloud"] = pillars("c", fed["Centralised cloud"]["acc"][0], fed["Centralised cloud"]["min_client"][0], float("inf"), 0.0,
                                   EXTRA["central_20"][0] / EXTRA["central_clean"][0], raw)
SYS["Local-only edge"] = pillars("l", fed["Local-only edge"]["acc"][0], fed["Local-only edge"]["min_client"][0], 0.0, 1.0,
                                 EXTRA["local_20"][0] / EXTRA["local_clean"][0], 0)
SYS["FedAvg"] = pillars("f", fed["FedAvg"]["acc"][0], fed["FedAvg"]["min_client"][0], float("inf"), 1.0,
                        rob["FedAvg"][i2][0] / rob["FedAvg"][i0][0], fed["FedAvg"]["comm"][0])
SYS["DP-FedAvg + SecAgg"] = pillars("d", fed["DP sigma=1.0"]["acc"][0], fed["DP sigma=1.0"]["min_client"][0], fed["DP sigma=1.0"]["eps"], 1.0,
                                    rob["DP sigma=1.0"][i2][0] / rob["DP sigma=1.0"][i0][0], fed["DP sigma=1.0"]["comm"][0])
WEIGHTS = {"Equal": np.ones(6) / 6, "Privacy-weighted": np.array([0.15, 0.2, 0.25, 0.1, 0.15, 0.15]),
           "Utility-weighted": np.array([0.35, 0.1, 0.1, 0.15, 0.15, 0.15])}
TS = {s: {w: float(np.dot(wt, list(p.values()))) for w, wt in WEIGHTS.items()} for s, p in SYS.items()}
DER["pillars"] = SYS; DER["trust"] = TS

# ---- Fig 25 trustworthiness (wide 1x3)
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.5))
a = ax[0]
for k, col, lab in (("FedAvg", C["dark"], "FedAvg (mean)"), ("Median", C["gov"], "Coordinate median"), ("DP sigma=1.0", C["cloud"], "DP-FedAvg \u03c3=1")):
    a.errorbar([f * 100 for f in rob["frac"]], [v[0] for v in rob[k]], yerr=[v[1] for v in rob[k]], marker="o", ms=3, lw=1, capsize=2, color=col, label=lab)
a.set_xlabel("Label-flipping clients (%)"); a.set_ylabel("Test accuracy"); a.set_title("(a) Robustness to poisoning", pad=3); a.legend(frameon=False)
a = ax[1]; n2 = RES["E2_noniid"]; xa = np.arange(len(n2["alpha"]))
a.plot(xa, [v[0] for v in n2["acc"]], marker="o", ms=3, color=C["edge"], label="Global accuracy")
a.plot(xa, [v[0] for v in n2["min_client"]], marker="s", ms=3, color=C["red"], label="Worst-client accuracy")
a.set_xticks(xa); a.set_xticklabels([f"{v:g}" for v in n2["alpha"]]); a.set_xlabel("Dirichlet \u03b1 (lower = more non-IID)"); a.set_ylabel("Accuracy")
a.set_title("(b) Effect of data heterogeneity", pad=3); a.legend(frameon=False)
a = ax[2]; names = list(TS.keys()); w = 0.26
for i, (wn, col) in enumerate((("Equal", C["edge"]), ("Privacy-weighted", C["gov"]), ("Utility-weighted", C["cloud"]))):
    a.bar(np.arange(len(names)) + (i - 1) * w, [TS[s][wn] for s in names], width=w, color=col, label=wn)
a.set_xticks(range(len(names))); a.set_xticklabels([n.replace(" + ", "\n+ ").replace(" cloud", "\ncloud").replace("-only edge", "-only\nedge") for n in names], fontsize=6)
a.set_ylabel("Trust score T"); a.set_ylim(0, 1); a.set_title("(c) Trust score by system", pad=3); a.legend(frameon=False, loc="upper right", fontsize=5.8)
fig.tight_layout(); save(fig, 25, "Trustworthiness Assessment Results")

# ---- Fig 26 radar
labels = ["Utility", "Data\nminimisation", "Formal\nprivacy", "Fairness", "Robustness", "Comm.\nefficiency"]
ang = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(3.45, 3.3)); ax = fig.add_subplot(111, polar=True)
for (s, p), col in zip(SYS.items(), (C["red"], C["gray"], C["edge"], C["gov"])):
    v = list(p.values()) + [list(p.values())[0]]; ax.plot(ang, v, color=col, lw=1.1, label=s); ax.fill(ang, v, color=col, alpha=0.06)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=6.5); ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=5.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=6)
save(fig, 26, "Comparative Radar Chart")

# ---- business model (all parameters are assumptions, not measurements)
N, MB_DAY, KEEP_MONTHS = 10000, 200.0, 6
GB_PRICE, STORE_PRICE = 0.05, 0.023
EDGE_UPLIFT, GOV_TOOLING, MODEL_MB, ROUNDS_YR = 12.0, 25000.0, 5.0, 52
P_BREACH, BREACH_COST, FEAT_RATIO = 0.03, 2.0e6, 0.10
raw_gb = N * MB_DAY / 1024 * 365
def bizcost(p_breach=P_BREACH, breach=BREACH_COST, price=GB_PRICE):
    cloud_net = raw_gb * price; cloud_store = raw_gb * STORE_PRICE * KEEP_MONTHS
    cloud_risk = p_breach * breach * 1.0
    off_gb = raw_gb * P_OFF * FEAT_RATIO; fl_gb = N * 2 * MODEL_MB / 1024 * ROUNDS_YR
    h_net = (off_gb + fl_gb) * price; h_store = off_gb * STORE_PRICE * KEEP_MONTHS
    h_edge = N * EDGE_UPLIFT; h_gov = GOV_TOOLING; exposure = P_OFF * FEAT_RATIO
    h_risk = p_breach * breach * exposure
    return dict(cloud=dict(network=cloud_net, storage=cloud_store, edge=0, gov=0, risk=cloud_risk),
                hybrid=dict(network=h_net, storage=h_store, edge=h_edge, gov=h_gov, risk=h_risk), exposure=exposure,
                raw_gb=raw_gb, off_gb=off_gb, fl_gb=fl_gb)
B0 = bizcost(); tot = lambda d: sum(d.values())
c_tot, h_tot = tot(B0["cloud"]), tot(B0["hybrid"]); incr = B0["hybrid"]["edge"] + B0["hybrid"]["gov"]
DER["biz"] = dict(B0=B0, cloud_total=c_tot, hybrid_total=h_tot, saving=c_tot - h_tot, saving_pct=(c_tot - h_tot) / c_tot * 100,
                  incremental_invest=incr, roi=(c_tot - h_tot + incr) / incr - 1 if incr else None,
                  net_benefit_ratio=(c_tot - h_tot) / incr, exposure=B0["exposure"])

fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.55))
a = ax[0]; comps = ["network", "storage", "edge", "gov"]; cols = [C["edge"], C["cloud"], C["gov"], C["sec"]]
bottoms = np.zeros(2)
for comp, col in zip(comps, cols):
    vals = np.array([B0["cloud"][comp], B0["hybrid"][comp]]) / 1e3
    a.bar([0, 1], vals, bottom=bottoms, color=col, width=0.55, label={"network": "Network", "storage": "Storage", "edge": "Edge upgrade", "gov": "Governance tooling"}[comp]); bottoms += vals
a.set_xticks([0, 1]); a.set_xticklabels(["Cloud-only", "Hybrid"]); a.set_ylabel("Annual infrastructure cost (k$)"); a.set_title("(a) Operating cost (assumed inputs)", pad=3); a.legend(frameon=False, fontsize=5.8)
a = ax[1]
a.bar([0, 1], [B0["cloud"]["risk"] / 1e3, B0["hybrid"]["risk"] / 1e3], color=[C["red"], C["gov"]], width=0.55)
a.set_xticks([0, 1]); a.set_xticklabels(["Cloud-only", "Hybrid"]); a.set_ylabel("Expected annual breach loss (k$)"); a.set_title("(b) Expected privacy-risk cost", pad=3)
for xi, v in enumerate([B0["cloud"]["risk"], B0["hybrid"]["risk"]]): a.text(xi, v / 1e3 + 1, f"{v/1e3:.1f}", ha="center", fontsize=6.5)
a = ax[2]; pb = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08]); bc = [1e6, 2e6, 4e6]
for b_, col in zip(bc, (C["gray"], C["edge"], C["cloud"])):
    net = [(tot(bizcost(p, b_)["cloud"]) - tot(bizcost(p, b_)["hybrid"])) / 1e3 for p in pb]
    a.plot(pb * 100, net, marker="o", ms=3, lw=1, color=col, label=f"Breach cost ${b_/1e6:.0f}M")
a.axhline(0, color="black", lw=0.6); a.set_xlabel("Annual breach probability (%)"); a.set_ylabel("Net annual benefit of hybrid (k$)")
a.set_title("(c) Sensitivity of net benefit", pad=3); a.legend(frameon=False)
fig.tight_layout(); save(fig, 30, "Business Impact Dashboard")
DER["biz"]["break_even_pb_at_2M"] = None
# break-even breach probability at 2M cost
for p in np.linspace(0.0001, 0.1, 2000):
    if tot(bizcost(p, 2e6)["cloud"]) - tot(bizcost(p, 2e6)["hybrid"]) >= 0:
        DER["biz"]["break_even_pb_at_2M"] = float(p); break

# ---- Fig 35 explanation maps (appendix, wide)
imp = RES["E6_explain"]["imp"]
fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.0))
for a, (k, v) in zip(ax, imp.items()):
    im = a.imshow(np.array(v).reshape(8, 8), cmap="viridis"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    rho = {"Centralised": 1.0, "FedAvg": RES["E6_explain"]["rho_fedavg_central"], "DP sigma=1.0": RES["E6_explain"]["rho_dp1_central"], "DP sigma=2.0": RES["E6_explain"]["rho_dp2_central"]}[k]
    name = k.replace('sigma', '\u03c3')
    rho_str = format(rho, '.2f')
    a.set_title(name + "\n\u03c1 vs centralised = " + rho_str, pad=3, fontsize=7)
fig.tight_layout(); save(fig, 35, "Pixel Importance Maps under Privacy")

json.dump(DER, open(os.path.join(BASE, "results", "derived.json"), "w"), indent=1, default=float)
json.dump({str(k): v for k, v in NAMES.items()}, open(os.path.join(BASE, "results", "figure_index.json"), "w"), indent=1)
print("figures:", len(NAMES))
print(json.dumps({k: DER[k] for k in ("latency_at_10mbps_ms", "throughput_at_10mbps", "latency_at_1mbps_ms", "offload_tau07", "trust")}, indent=1))
print(json.dumps({k: v for k, v in DER["biz"].items() if k != "B0"}, indent=1))
print({k: round(sum(v.values())) for k, v in DER["biz"]["B0"].items() if isinstance(v, dict)}, DER["biz"]["B0"]["raw_gb"], DER["biz"]["B0"]["off_gb"], DER["biz"]["B0"]["fl_gb"])
print(json.dumps(DER["pillars"], indent=1))
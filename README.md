# Privacy-Aware Cloud–Edge Intelligence for Distributed Data Science

Case-study package for **BCSE318L Data Privacy**, School of Computer Science and Engineering, Vellore Institute of Technology (Team ID 26FSDPYCPMID1030).

Authors: Lakshay Chauhan (23BCE0644), Suraj Goyal (23BCE0110), Shrish Shrinivas K (23BCE0031).
Guide: Dr. M. Rajasekhara Babu.

## Deliverables

| Folder / file | Contents |
|---|---|
| `Privacy-Aware_Cloud-Edge_Intelligence_Case_Study.docx` | Complete manuscript (Word format) |
| `01_Diagrams/` | Fig 01 – Fig 35 as 600 dpi PNG files, named `Fig NN <Figure Name>.png` |
| `01_Diagrams/drawio_sources/` | Editable draw.io sources for the 29 architecture and workflow diagrams |
| `01_Diagrams/dot_sources/` | Graphviz sources used to generate the same diagrams |
| `02_Source_Code/` | `fl_core.py` (library), `run_experiments.py`, `run_extra.py`, `make_figures.py`, `requirements.txt`; `preliminary_tuning/` holds the exploratory runs used to choose learning rates |
| `03_Datasets/` | Source datasets (`digits.csv`, `breast_cancer.csv`) and all generated results (`results.json`, `derived.json`, `extra_robustness.json`, `results_summary_E1.csv`, `figure_index.json`) |
| `04_Execution_Logs_and_Videos/` | `execution_log.txt` (console output of the run used for the report) and the execution video |

Figures 23–26, 30 and 35 are data charts produced directly by `make_figures.py` from the results files, so they have no draw.io source.

## Data availability

The datasets generated and/or analysed during the current study are available in this GitHub repository, https://github.com/lakshay-chauhan/data-privacy-project. Digits and Breast Cancer Wisconsin (Diagnostic) are public datasets distributed with scikit-learn; no private or patient-level data were used.

## Reproduce

Requirements: Python 3.10 or later and Graphviz (`dot` on the PATH; needed only for `make_figures.py`).

```
pip install -r 02_Source_Code/requirements.txt
cd 02_Source_Code
python run_experiments.py     # writes ../03_Datasets/results.json and results_summary_E1.csv
python run_extra.py           # writes ../03_Datasets/extra_robustness.json
python make_figures.py        # writes ../01_Diagrams/*.png and ../03_Datasets/derived.json
```

All randomness is seeded (seeds 0, 1, 2), so a rerun reproduces every accuracy, privacy-budget, and attack result exactly. Timing measurements (round time, inference latency) depend on the machine and its load and will differ slightly between runs.

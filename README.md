# Privacy-Aware Cloud–Edge Intelligence for Distributed Data Science — BCSE318L case study package

| Folder / file | What it is |
|---|---|
| `Privacy-Aware_Cloud-Edge_Intelligence_Case_Study.docx` | Complete manuscript (cover, index, lists, contributions, IEEE-style two-column paper, references, appendix) |
| `..._preview.pdf` | PDF rendering for quick reading (made with LibreOffice; Word layout may differ slightly) |
| `01_Diagrams/` | Fig 01–Fig 35 as 600 dpi PNG; `dot_sources/` holds the Graphviz source of each diagram |
| `02_Source_Code/` | `fl_core.py` (library), `run_experiments.py`, `run_extra.py`, `make_figures.py`, `requirements.txt`; `preliminary_tuning/` holds the exploratory runs used to pick learning rates |
| `03_Datasets/` | `digits.csv`, `breast_cancer.csv` (public scikit-learn datasets), raw results (`results.json`, `derived.json`, `extra_robustness.json`), `results_summary_E1.csv` |
| `04_Execution_Logs_and_Videos/` | `execution_log.txt`; execution videos are NOT included (see note inside) |
| `References_Verification_Tracker.csv` | All 83 references with category and verification status |

## Reproduce
```
pip install -r requirements.txt
cd 02_Source_Code
python run_experiments.py      # ~ a few minutes, writes ../results/results.json
python run_extra.py
python make_figures.py         # needs the Graphviz `dot` binary
```
The scripts create `results/` (and `diagrams/` for new figures) next to `02_Source_Code/` automatically. All randomness is seeded (seeds 0, 1, 2); a rerun reproduces the saved numbers exactly.

## Fill in before submission
Team ID, team member names and registration numbers (cover, author line, contributions table), VIT logo / guide photo if required, GitHub link in the Data and Code Availability section, execution videos.

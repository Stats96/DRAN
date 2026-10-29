# DRAN: DRAEN-Efron-SA for survival prediction from breast cancer transcriptomes

Code for the manuscript *Attention-gated residual deep Cox network with Efron partial likelihood improves survival
prediction from breast cancer transcriptomes* (Delino, Mardani-Fard, Ghaderi Zefrehee).

`draen_pipeline.py` implements the whole analysis in one file:

| Command | What it does |
|---|---|
| `sim` | Simulation study: ridge-Cox, random survival forest and DRAEN-Efron-SA, 3 scenarios x 200 replicates |
| `real` | SCAN-B (GSE96058) cross-validated comparison with in-fold gene screening and scaling, patient-grouped common folds, inner validation split for early stopping, C-index, time-dependent AUC and Brier score, gradient x input importance |
| `report` | Aggregates results into LaTeX tables, CSV summaries, paired tests and figures |

## Install
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data
Download GSE96058 from GEO. The input table has one row per RNA-seq profile with columns `OS_days`, `OS_event`, the 11
clinical covariates (`age_at_diagnosis, tumor_size_mm, lymph_node_status, er_status, pgr_status, her2_status,
ki67_status, nhg_grade, pam50_subtype, endocrine_treated, chemo_treated`), a patient-id column (so that the 136 technical
replicates stay in one fold) and one column per gene.

### Coding of the clinical covariates (as used in the paper)
`lymph_node_status`: 0 missing, 1 node-negative, 2 node-positive; `nhg_grade`: 0 missing, 1-3 grade;
`pam50_subtype`: 0 basal-like, 1 HER2-enriched, 2 luminal A, 3 luminal B, 4 normal-like;
`er_status`, `pgr_status`, `her2_status`, `ki67_status`: 0 negative/low, 1 positive/high, missing coded with an intermediate value;
`age_at_diagnosis`, `tumor_size_mm`: scaled values; `endocrine_treated`, `chemo_treated`: 0/1.

## Run
```
python draen_pipeline.py real --data full_dataset.tsv --id_col patient_id --out results_real      # add --smoke for a quick test
python draen_pipeline.py sim  --out results_sim
python draen_pipeline.py report --real_dir results_real --sim_dir results_sim
```
Optional extra models for `--models`: `clinical`, `mlp`, `draen_noattn`, `draen_nores`, `draen_single`.
Runs are seeded (default 42) and resumable. Library versions are written to `environment.txt` in the output folder.
A GPU is recommended: the full analysis trains 5 networks x 10 folds x 3 feature sets.

## Exact Efron loss
`efron_loss` is a vectorised Efron negative log partial likelihood and was checked against the Efron log-likelihood of
`lifelines` (agreement to six digits).

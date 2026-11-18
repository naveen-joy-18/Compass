# Compass / CRLP Reproducibility Guide

This repository implements a staged, evidence-controlled pipeline for:

- aligning DepMap CRISPR and expression matrices;
- curating experimentally supported compensation labels;
- generating candidate gene-pair features;
- organizing and profiling GSE90546 single-cell perturbation data;
- training nine representation-learning models; and
- performing supervised downstream validation with target-gene-grouped cross-validation.

The pipeline deliberately separates candidate discovery from experimental ground truth. Candidate pairs and descriptive perturbation profiles are **not** compensation labels.

## Scientific scope and limitations

- `8_train_all_nine_models.py` evaluates representations on the study's perturbation-response proxy. It does not train a gene-pair compensation classifier.
- GSE90546 pseudobulk contrasts are descriptive only. The supplied code does not claim inferential p-values or compensation.
- Script `2_define_compensation_labels.py` creates labels only from curator-entered experimental evidence.
- A label of `1` requires induced response, functional relationship, and compensation/rescue evidence.
- A label of `0` requires an adequate negative test and observed absence of compensation.
- Records not meeting one of those rules remain `-1` / uncertain and are excluded from supervised training.
- Do not infer negative labels from untested pairs, expression correlations, or perturbation clusters.

## Repository layout

| Path | Purpose |
|---|---|
| `1_preprocess.py` | Align DepMap matrices and metadata |
| `2_define_compensation_labels.py` | Curate and classify evidence-based labels |
| `3_feature_engineering.py` | Build gene features and candidate pair features |
| `4_organize_and_audit_GSE90546.py` | Copy, organize, and QC GSE90546 raw files |
| `5_align_and_profile_GSE90546.py` | Align cell barcodes to guide identities |
| `6_build_perturbation_profiles.py` | Create guide pseudobulks and descriptive contrasts |
| `7_audit_GSE90546_guide_mappings.py` | Additional guide-mapping audit utility |
| `8_train_all_nine_models.py` | Train and compare nine representation models |
| `9_biological_validation_feature_importance_ablation.py` | Supervised pair validation, feature importance, and ablation |
| `preprocessed/` | Aligned matrices, label tables, and QC reports |
| `features/` | Gene-level and candidate-pair feature tables |
| `perturbation_analysis/` | GSE90546 pseudobulk profiles and reports |
| `training_all_models/` | Nine-model metrics, embeddings, losses, and checkpoints |
| `downstream_validation/` | Curated downstream predictions and analyses |

## Environment

The analyzed snapshot was produced on Windows with Python 3.11.0. The following package versions were available in that environment:

```text
pandas==3.0.5
numpy==2.4.6
scipy==1.17.1
scikit-learn==1.9.0
torch==2.13.0+cpu
matplotlib==3.10.9
tqdm==4.70.0
```

Create an isolated environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn torch matplotlib tqdm psutil shap
python --version
python -m pip freeze
```

`psutil` is optional; `shap` is optional for downstream SHAP analysis. Record `python --version`, `pip freeze`, operating system, CPU/GPU details, and the date of rerun with any published result.

No CUDA build was used for the archived run. With a compatible CUDA build of PyTorch, GPU execution is supported automatically; the script will report the active device and memory usage.

## Required external data

### 1. DepMap data

Place the source files in `datasets/`:

```text
datasets/
  CRISPRGeneEffect.csv
  OmicsExpressionProteinCodingGenesTPMLogp1.csv
  Model.csv
```

Record the original DepMap/CCLE release identifier with the analysis. The exact release, download date, and any local modifications must be included in a reproducible report. The scripts expect the first column to contain model identifiers and subsequent columns to contain gene names.

### 2. GSE90546 data

The scripts expect three extracted GEO sample sets under:

```text
datasets/GSE90546_RAW/
  GSM2406675_10X001_barcodes.tsv.gz
  GSM2406675_10X001_cell_identities.csv.gz
  GSM2406675_10X001_genes.tsv.gz
  GSM2406675_10X001_matrix.mtx.txt.gz
  GSM2406677_10X005_barcodes.tsv.gz
  GSM2406677_10X005_cell_identities.csv.gz
  GSM2406677_10X005_genes.tsv.gz
  GSM2406677_10X005_matrix.mtx.txt.gz
  GSM2406681_10X010_barcodes.tsv.gz
  GSM2406681_10X010_cell_identities.csv.gz
  GSM2406681_10X010_genes.tsv.gz
  GSM2406681_10X010_matrix.mtx.txt.gz
```

Download and extract the original GSE90546 supplementary files yourself. Script `4_organize_and_audit_GSE90546.py` currently contains a Windows-specific source path:

```python
SOURCE_DIR = Path(r"C:\Users\Naveen Joy\Downloads\GSE90546_RAW")
```

Edit this value to the extracted GSE90546 source directory before running script `4`.

## Pipeline execution

Run commands from the repository root.

### A. DepMap preprocessing and feature engineering

```powershell
python 1_preprocess.py
python 2_define_compensation_labels.py
```

For script `2`, the first run creates:

```text
preprocessed/compensation_evidence_template.csv
```

Fill in independent experimental evidence and save the completed file as:

```text
preprocessed/compensation_evidence_annotated.csv
```

Then rerun script `2` to generate:

```text
preprocessed/compensation_labels_all.csv
preprocessed/compensation_training_pairs.csv
preprocessed/compensation_label_QC.json
```

Create candidate features after preprocessing:

```powershell
python 3_feature_engineering.py
```

### B. GSE90546 preparation

Edit the source path in script `4`, then run:

```powershell
python 4_organize_and_audit_GSE90546.py
python 5_align_and_profile_GSE90546.py
python 6_build_perturbation_profiles.py
```

The first invocation of script `6` creates:

```text
preprocessed/GSE90546_QC/guide_reference.csv
```

Review every guide mapping against the original study guide design, then set `mapping_verified=TRUE` only for mappings confirmed by source documentation. Set `is_negative_control=TRUE` or `FALSE` for each verified row and blank `target_symbols` for negative controls.

After review:

```powershell
python 6_build_perturbation_profiles.py profiles
```

Script `7_audit_GSE90546_guide_mappings.py` is an additional audit implementation. Use its output for mapping review; it is not a substitute for verification against the original publication or supplementary guide table.

### C. Nine-model representation suite

```powershell
python 8_train_all_nine_models.py `
  --epochs 100 `
  --top-genes 256 `
  --latent-dim 16 `
  --batch-size 16 `
  --lr 0.001 `
  --val-ratio 0.20 `
  --seed 42
```

Defaults:

```text
epochs=100
top-genes=256
latent-dim=16
batch-size=16
learning-rate=0.001
validation-ratio=0.20
seed=42
```

For a faster smoke run:

```powershell
python 8_train_all_nine_models.py --epochs 5 --top-genes 64 --seed 42
```

Smoke-run outputs are not equivalent to the archived 100-epoch results.

### D. Supervised biological validation

The label file must contain both positive and negative evidence and must overlap with at least 20 candidate pairs.

```powershell
python 9_biological_validation_feature_importance_ablation.py `
  --top-k 20 `
  --folds 5 `
  --seed 42
```

Disable SHAP if unavailable or not needed:

```powershell
python 9_biological_validation_feature_importance_ablation.py --no-shap
```

## Random seeds and deterministic settings

The principal seeds are:

- representation suite: `42`
- downstream grouped validation: `42`

Script `8` seeds Python, NumPy, and PyTorch. CUDA operations may still have hardware-dependent nondeterminism. Script `9` uses the seed for repeated folds and bootstrap-style analyses where applicable.

For more deterministic PyTorch execution, also record the following environment variables in the run command:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONHASHSEED = "42"
```

PyTorch does not guarantee bitwise identical results across different versions or hardware.

## Expected outputs

### Preprocessing

```text
preprocessed/CRISPRGeneEffect_matched.csv.gz
preprocessed/Expression_matched.csv.gz
preprocessed/Model_metadata_matched.csv
preprocessed/common_genes.txt
preprocessed/preprocessing_QC.json
```

### Labels

```text
preprocessed/compensation_evidence_template.csv
preprocessed/compensation_evidence_annotated.csv
preprocessed/compensation_labels_all.csv
preprocessed/compensation_training_pairs.csv
preprocessed/compensation_label_QC.json
```

### Candidate features

```text
features/gene_level_features.csv
features/candidate_gene_pairs.csv.gz
features/feature_engineering_QC.json
```

### GSE90546

```text
preprocessed/GSE90546_QC/
perturbation_analysis/GSE90546_guide_pseudobulk_profiles.csv.gz
perturbation_analysis/GSE90546_pseudobulk_sample_summary.csv
perturbation_analysis/pretraining_data_report.json
perturbation_analysis/unverified_guides.csv
```

### Representation models

```text
training_all_models/all_models_summary.csv
training_all_models/all_models_classification_metrics.csv
training_all_models/all_models_reconstruction_metrics.csv
training_all_models/all_models_diagnostics.csv
training_all_models/all_models_training_report.json
training_all_models/models/*.pt
```

### Downstream validation

```text
downstream_validation/all_ranked_candidate_predictions.csv
downstream_validation/grouped_oof_metrics.json
downstream_validation/grouped_cv_metrics_by_fold.csv
downstream_validation/grouped_oof_pair_predictions.csv
downstream_validation/permutation_feature_importance.csv
downstream_validation/feature_group_ablation.csv
downstream_validation/feature_group_importance.csv
downstream_validation/known_compensation_topk_validation.csv
downstream_validation/top_novel_computational_hypotheses.csv
downstream_validation/analysis_availability_and_scope.json
downstream_validation/figures/
```

## QC and verification

At minimum, preserve and inspect:

- `preprocessing_QC.json`
- `compensation_label_QC.json`
- `feature_engineering_QC.json`
- `preprocessed/GSE90546_QC/GSE90546_import_QC.json`
- `preprocessed/GSE90546_QC/alignment/alignment_and_guide_QC.json`
- `perturbation_analysis/pretraining_data_report.json`
- `training_all_models/all_models_training_report.json`
- `downstream_validation/analysis_availability_and_scope.json`

Do not interpret output as reproducible until the rerun uses the same source releases and mapping decisions, has both curated label classes, and records any differences in package versions or hardware.

## Optional manual evidence

For extended downstream validation, scripts can consume:

```text
preprocessed/gene_function_terms.csv
  columns: gene, term

preprocessed/external_validation_pairs.csv
  columns: target_gene, candidate_gene, label, plus feature columns

preprocessed/celltype_predictions.csv
  columns: target_gene, candidate_gene, cell_type, prediction_score
```

These files are optional. Their absence is reported in `analysis_availability_and_scope.json`; it must not be treated as a positive or negative validation result.

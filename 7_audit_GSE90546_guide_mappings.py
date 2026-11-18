"""
CRLP pre-training data pipeline for GSE90546.

Run from the BA project directory:
    python 6_prepare_guide_reference.py
    # Review and edit preprocessed/GSE90546_QC/guide_reference.csv
    python 7_build_perturbation_profiles.py

Scientific scope:
- Builds cell-to-guide mappings and per-guide pseudobulk profiles.
- Computes descriptive perturbation-vs-control log2 fold changes.
- Does NOT claim functional compensation.
- Does NOT generate inferential p-values from cells as if they were biological replicates.
- Does NOT create negative compensation labels from untested gene pairs.
"""

from pathlib import Path
import gzip
import json
import re
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "GSE90546_RAW"
QC = ROOT / "preprocessed" / "GSE90546_QC"
ALIGN = QC / "alignment"
OUT = ROOT / "perturbation_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "GSM2406675_10X001",
    "GSM2406677_10X005",
    "GSM2406681_10X010",
]

GUIDE_REF = QC / "guide_reference.csv"
MIN_CELLS_PER_GUIDE = 30
PSEUDOCOUNT = 0.1

def read_gz_lines(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return [x.rstrip("\r\n") for x in f]

def load_sparse_mtx(path):
    with gzip.open(path, "rt", encoding="ascii") as f:
        mat = mmread(f)
    if not sparse.issparse(mat):
        mat = sparse.coo_matrix(mat)
    return mat.tocsc()

def sample_guide_ids(sample):
    p = ALIGN / f"{sample}_cell_to_guide.csv.gz"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run 5_align_and_profile_GSE90546.py first."
        )
    df = pd.read_csv(p, compression="gzip", low_memory=False)
    return sorted(df["guide_identity"].dropna().astype(str).unique())

def make_reference_template():
    rows = []
    for sample in SAMPLES:
        for guide in sample_guide_ids(sample):
            target = ""
            guide_type = "unknown"
            is_control = ""
            # Candidate symbol prefix only; mapping still needs verification.
            if "_pDS" in guide:
                target = guide.split("_pDS")[0]
                guide_type = "single_candidate"
            elif guide in {"ATF6_only_pMJ145"}:
                target, guide_type = "ATF6", "single_candidate"
            elif guide in {"PERK_only_pMJ146"}:
                target, guide_type = "EIF2AK3", "single_candidate"
            elif guide in {"IRE1_only_pMJ148"}:
                target, guide_type = "ERN1", "single_candidate"
            elif guide == "ATF6_PERK_pMJ150":
                target, guide_type = "ATF6;EIF2AK3", "combinatorial_candidate"
            elif guide == "ATF6_IRE1_pMJ152":
                target, guide_type = "ATF6;ERN1", "combinatorial_candidate"
            elif guide == "PERK_IRE1_pMJ154":
                target, guide_type = "EIF2AK3;ERN1", "combinatorial_candidate"
            elif guide == "ATF6_PERK_IRE1_pMJ158":
                target, guide_type = "ATF6;EIF2AK3;ERN1", "combinatorial_candidate"
            elif "neg_ctrl" in guide.lower():
                guide_type, is_control = "negative_control_candidate", "TRUE"
            elif guide == "*":
                guide_type = "unassigned"
            else:
                guide_type = "unverified_special"

            rows.append({
                "sample_id": sample,
                "guide_identity": guide,
                "target_symbols": target,
                "guide_type": guide_type,
                "is_negative_control": is_control,
                "mapping_verified": "FALSE",
                "notes": "Verify against original GEO/paper supplementary guide table."
            })

    df = pd.DataFrame(rows).drop_duplicates(["sample_id", "guide_identity"])
    # The explicit negative-control naming is a strong annotation, but still
    # requires source verification before the mapping is accepted.
    df.to_csv(GUIDE_REF, index=False)
    print(f"Created guide-reference review template: {GUIDE_REF}")
    print(f"Rows requiring review: {len(df)}")
    print("\nReview every row. In particular:")
    print("- Confirm target_symbols against the original guide design.")
    print("- Confirm negative controls from original study documentation.")
    print("- Set mapping_verified=TRUE only after checking the source.")
    print("- For negative controls, set is_negative_control=TRUE.")
    print("- For perturbations, set is_negative_control=FALSE.")
    print("- Keep unassigned/uncertain guides out of the analysis.")
    print("\nThen run this script again with argument 'profiles'.")
    return df

def validate_reference(ref):
    """
    Validate schema and consistency without requiring guide-to-target verification.
    Unverified rows are retained for GUIDE-IDENTITY exploratory profiling only.
    """
    required = {
        "sample_id", "guide_identity", "target_symbols", "guide_type",
        "is_negative_control", "mapping_verified"
    }
    missing = required - set(ref.columns)
    if missing:
        raise ValueError(f"Guide reference is missing columns: {sorted(missing)}")

    ref = ref.copy()
    for col in ["sample_id", "guide_identity", "target_symbols", "guide_type",
                "is_negative_control", "mapping_verified"]:
        ref[col] = ref[col].fillna("").astype(str).str.strip()

    ref["mapping_verified"] = ref["mapping_verified"].str.upper()
    ref["is_negative_control"] = ref["is_negative_control"].str.upper()

    if not ref["mapping_verified"].isin(["TRUE", "FALSE", ""]).all():
        raise ValueError("mapping_verified must contain TRUE, FALSE, or blank.")

    # A guide may occur in different samples, but only once within each sample.
    dup = ref.duplicated(["sample_id", "guide_identity"], keep=False)
    if dup.any():
        bad = ref.loc[dup, ["sample_id", "guide_identity"]]
        raise ValueError("Duplicate sample/guide rows in guide_reference.csv:\\n"
                         + bad.to_string(index=False))

    # Only explicitly verified control annotations can be used for contrasts.
    # Unverified candidate controls remain in the profiles but are not treated
    # as controls.
    return ref

def make_profiles():
    if not GUIDE_REF.exists():
        make_reference_template()
        raise SystemExit(
            "Guide-reference template created. Verify it before profile generation."
        )

    ref = pd.read_csv(GUIDE_REF, dtype=str, keep_default_na=False)
    ref = validate_reference(ref)

    profile_summaries = []
    de_summaries = []
    all_profiles = []

    for sample in SAMPLES:
        print(f"\nProcessing {sample}")
        bpath = DATA / f"{sample}_barcodes.tsv.gz"
        gpath = DATA / f"{sample}_genes.tsv.gz"
        mpath = DATA / f"{sample}_matrix.mtx.txt.gz"
        map_path = ALIGN / f"{sample}_cell_to_guide.csv.gz"

        barcodes = read_gz_lines(bpath)
        genes = pd.read_csv(
            gpath, sep="\t", header=None, compression="gzip",
            names=["gene_id", "gene_symbol"], dtype=str
        )
        mapping = pd.read_csv(map_path, compression="gzip", low_memory=False)
        mapping["cell_barcode"] = mapping["cell_barcode"].astype(str)
        mapping = mapping[mapping["annotation_match"] == True].copy()

        ref_s = ref[ref["sample_id"] == sample].copy()
        mapping = mapping.merge(
            ref_s[["guide_identity", "target_symbols", "guide_type",
                   "is_negative_control", "mapping_verified"]],
            left_on="guide_identity", right_on="guide_identity",
            how="left", validate="many_to_one"
        )
        # Retain guide identities even when guide-to-target mapping is unresolved.
        # Such rows are usable for exploratory guide-identity profiling, but not
        # for target-gene claims or control contrasts.
        mapping["mapping_verified"] = (
            mapping["mapping_verified"].fillna("").astype(str).str.upper()
        )
        mapping["target_symbols"] = mapping["target_symbols"].fillna("").astype(str)
        mapping["guide_type"] = mapping["guide_type"].fillna("unverified").astype(str)
        mapping["is_negative_control"] = (
            mapping["is_negative_control"].fillna("").astype(str).str.upper()
        )
        mapping["mapping_status"] = np.where(
            mapping["mapping_verified"] == "TRUE", "verified", "unverified"
        )
        mapping = mapping[mapping["guide_identity"].notna()].copy()

        if mapping.empty:
            print(f"  No guide-annotated cells for {sample}; skipping.")
            continue

        # Use barcode-to-column matching; never assume annotation row order.
        barcode_to_col = {bc: i for i, bc in enumerate(barcodes)}
        mapping["matrix_col_index"] = mapping["cell_barcode"].map(barcode_to_col)
        mapping = mapping.dropna(subset=["matrix_col_index"])
        mapping["matrix_col_index"] = mapping["matrix_col_index"].astype(int)

        matrix = load_sparse_mtx(mpath)
        if matrix.shape != (len(genes), len(barcodes)):
            raise ValueError(f"{sample}: matrix dimensions do not match genes/barcodes.")

        # Aggregate raw counts by verified guide identity (pseudobulk).
        for guide, group in mapping.groupby("guide_identity"):
            cols = group["matrix_col_index"].to_numpy()
            if len(cols) < MIN_CELLS_PER_GUIDE:
                print(f"  Skipping {guide}: only {len(cols)} matched cells.")
                continue

            summed = np.asarray(matrix[:, cols].sum(axis=1)).ravel()
            libsize = float(summed.sum())
            if libsize <= 0:
                continue
            cpm = summed / libsize * 1e6

            meta = group.iloc[0]
            profile = pd.DataFrame({
                "gene_id": genes["gene_id"],
                "gene_symbol": genes["gene_symbol"],
                "sample_id": sample,
                "guide_identity": guide,
                "target_symbols": meta["target_symbols"],
                "guide_type": meta["guide_type"],
                "is_negative_control": meta["is_negative_control"],
                "mapping_verified": meta["mapping_verified"],
                "mapping_status": meta["mapping_status"],
                "n_cells": int(len(cols)),
                "sum_counts": summed.astype(np.int64),
                "CPM": cpm,
                "log2_CPM": np.log2(cpm + PSEUDOCOUNT),
            })
            all_profiles.append(profile)
            profile_summaries.append({
                "sample_id": sample,
                "guide_identity": guide,
                "target_symbols": meta["target_symbols"],
                "guide_type": meta["guide_type"],
                "is_negative_control": meta["is_negative_control"],
                "mapping_verified": meta["mapping_verified"],
                "mapping_status": meta["mapping_status"],
                "n_cells": int(len(cols)),
                "library_size": int(summed.sum())
            })
            print(f"  {guide}: {len(cols):,} cells, library={int(libsize):,}")

        del matrix

    if not all_profiles:
        raise RuntimeError(
            "No pseudobulk profiles were generated. Check verified guide mapping "
            "and minimum cell thresholds."
        )

    profiles = pd.concat(all_profiles, ignore_index=True)
    profiles.to_csv(OUT / "GSE90546_guide_pseudobulk_profiles.csv.gz",
                    index=False, compression="gzip")
    pd.DataFrame(profile_summaries).to_csv(
        OUT / "GSE90546_pseudobulk_sample_summary.csv", index=False
    )

    # Descriptive contrasts: each perturbation against the pooled negative
    # control(s) from the SAME sample. No inferential p-values are claimed.
    profile_key = ["sample_id", "guide_identity"]
    wide = profiles.pivot_table(
        index=["sample_id", "gene_id", "gene_symbol"],
        columns="guide_identity", values="log2_CPM", aggfunc="first"
    )
    meta = profiles.drop_duplicates(profile_key).set_index(profile_key)

    contrast_rows = []
    for sample in sorted(profiles["sample_id"].unique()):
        sample_meta = meta.loc[sample]
        control_guides = sample_meta[
            (sample_meta["is_negative_control"].astype(str).str.upper() == "TRUE")
            & (sample_meta["mapping_verified"].astype(str).str.upper() == "TRUE")
        ].index.tolist()
        if not control_guides:
            print(f"WARNING: no verified negative controls for {sample}; no contrasts.")
            continue

        # Control mean across explicitly verified negative-control guide profiles.
        control_cols = [g for g in control_guides if g in wide.columns]
        control_mean = wide.xs(sample, level=0)[control_cols].mean(axis=1)

        perturb_rows = sample_meta[
            (sample_meta["is_negative_control"].astype(str).str.upper() == "FALSE")
            & (sample_meta["mapping_verified"].astype(str).str.upper() == "TRUE")
        ]
        for guide in perturb_rows.index:
            if guide not in wide.columns:
                continue
            pert = wide.xs(sample, level=0)[guide]
            lfc = pert - control_mean
            out = lfc.rename("log2FC_vs_control").reset_index()
            out["sample_id"] = sample
            out["guide_identity"] = guide
            out["target_symbols"] = perturb_rows.loc[guide, "target_symbols"]
            out["n_cells_perturbation"] = int(perturb_rows.loc[guide].get("n_cells", 0))
            out["n_control_guides"] = len(control_cols)
            out["contrast_type"] = "descriptive_pseudobulk_log2CPM_difference"
            out["p_value"] = np.nan
            out["adjusted_p_value"] = np.nan
            contrast_rows.append(out)

    if contrast_rows:
        contrasts = pd.concat(contrast_rows, ignore_index=True)
        contrasts.to_csv(
            OUT / "GSE90546_descriptive_perturbation_response.csv.gz",
            index=False, compression="gzip"
        )
    else:
        contrasts = pd.DataFrame()

    report = {
        "pseudobulk_profiles": int(len(profile_summaries)),
        "profile_rows_gene_by_guide": int(len(profiles)),
        "descriptive_contrast_rows": int(len(contrasts)),
        "samples_with_profiles": sorted(profiles["sample_id"].unique().tolist()),
        "samples_with_control_contrasts": sorted(
            contrasts["sample_id"].unique().tolist()
        ) if len(contrasts) else [],
        "statistical_limitation": (
            "This archive provides one pooled experiment per sample. Cell-level "
            "counts are not independent biological replicates. Profiles from "
            "unverified guide identities are exploratory only. Contrasts are "
            "generated only when both perturbation and negative-control mappings "
            "are explicitly verified. No inferential p-values, FDR values, or "
            "functional compensation labels are generated."
        ),
        "supervised_training_warning": (
            "Do not treat each guide identity as a supervised class with one "
            "pseudobulk profile. A single profile per guide does not provide "
            "independent within-class training/test examples. Cell-level random "
            "splits would risk leakage and pseudoreplication. Additional "
            "biological replicates or independent perturbation experiments are "
            "needed for defensible supervised generalization."
        ),
        "next_requirements": [
            "Resolve guide-to-target mappings from original supplementary guide annotations.",
            "Verify negative controls before interpreting perturbation-versus-control contrasts.",
            "Determine whether independent biological replicates exist in the study design.",
            "Use replicate-aware pseudobulk differential expression if replicate data exist.",
            "Obtain independent functional evidence before assigning compensation labels.",
            "Do not train a supervised guide classifier from one pseudobulk profile per guide."
        ]
    }
    with open(OUT / "pretraining_data_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nPRETRAINING PROFILE GENERATION FINISHED")
    print(json.dumps(report, indent=2))
    print(f"\nOutputs: {OUT}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        make_reference_template()
    elif sys.argv[1].lower() == "profiles":
        make_profiles()
    else:
        print("Usage:")
        print("  python 7_build_perturbation_profiles.py")
        print("  python 7_build_perturbation_profiles.py profiles")

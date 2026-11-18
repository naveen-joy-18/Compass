"""
5_align_and_profile_GSE90546.py

Purpose:
- Align each sample's cell identity annotations to its matrix barcodes.
- Diagnose the 10/150/80 row discrepancies found in the initial QC.
- Inventory ALL guide identities and their cell counts.
- Classify guide strings provisionally as control / single / combination / unclear.
- Save cell-to-guide mapping and gene annotations for downstream pseudobulk analysis.

No dense matrix conversion. No differential-expression labels are generated here.
"""

from pathlib import Path
import gzip
import json
import re
import warnings

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "GSE90546_RAW"
OUT = ROOT / "preprocessed" / "GSE90546_QC" / "alignment"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "GSM2406675_10X001",
    "GSM2406677_10X005",
    "GSM2406681_10X010",
]

def read_gz_lines(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return [x.rstrip("\r\n") for x in f]

def read_matrix(path):
    with gzip.open(path, "rt", encoding="ascii") as f:
        m = mmread(f)
    if not sparse.issparse(m):
        m = sparse.coo_matrix(m)
    return m.tocsc()

def classify_guide(guide):
    s = str(guide).strip()
    low = s.lower()
    if not s or low in {"nan", "none", "unknown", "unassigned"}:
        return "unassigned"
    if any(x in low for x in ["neg_ctrl", "negative_control", "non_target", "ntc"]):
        return "control_candidate"
    # The known branch-combination labels contain multiple target names.
    if any(x in s for x in ["ATF6_PERK", "ATF6_IRE1", "PERK_IRE1"]) or s.count("_") >= 2:
        # Do not rely on underscore count alone to decide biology.
        if any(x in s for x in ["ATF6_PERK", "ATF6_IRE1", "PERK_IRE1"]):
            return "combinatorial_candidate"
    return "single_or_other"

def main():
    sample_reports = []
    all_guide_counts = []
    all_mappings = []
    all_gene_annotations = []

    for sample in SAMPLES:
        print("\n" + "=" * 72)
        print("Processing", sample)
        print("=" * 72)

        bpath = DATA / f"{sample}_barcodes.tsv.gz"
        ipath = DATA / f"{sample}_cell_identities.csv.gz"
        gpath = DATA / f"{sample}_genes.tsv.gz"
        mpath = DATA / f"{sample}_matrix.mtx.txt.gz"

        for p in [bpath, ipath, gpath, mpath]:
            if not p.exists():
                raise FileNotFoundError(p)

        barcodes = read_gz_lines(bpath)
        identity = pd.read_csv(ipath, compression="gzip", low_memory=False)
        genes = pd.read_csv(
            gpath, sep="\t", header=None, compression="gzip",
            names=["gene_id", "gene_name"], dtype=str
        )
        matrix = read_matrix(mpath)

        required = {"cell BC", "guide identity"}
        if not required.issubset(identity.columns):
            raise ValueError(
                f"{sample}: expected columns {required}; got {list(identity.columns)}"
            )

        if matrix.shape != (len(genes), len(barcodes)):
            raise ValueError(
                f"{sample}: matrix shape {matrix.shape} does not equal "
                f"(genes={len(genes)}, barcodes={len(barcodes)})"
            )

        identity["cell BC"] = identity["cell BC"].astype(str).str.strip()
        identity["guide identity"] = identity["guide identity"].astype(str).str.strip()

        # Diagnose duplicates before joining.
        duplicate_matrix_barcodes = pd.Series(barcodes).duplicated().sum()
        duplicate_identity_barcodes = identity["cell BC"].duplicated().sum()

        # Matrix barcode -> column index. Preserve matrix order.
        barcode_map = pd.DataFrame({
            "cell_barcode": barcodes,
            "matrix_col_index": np.arange(len(barcodes), dtype=np.int64)
        })

        # Identity barcode -> guide. Do not join by row position.
        identity_small = identity.copy()
        identity_small = identity_small.rename(columns={
            "cell BC": "cell_barcode",
            "guide identity": "guide_identity"
        })

        # If repeated annotation rows exist for a barcode, keep a record and
        # refuse ambiguous conflicting guide assignments.
        conflicting = (
            identity_small.groupby("cell_barcode")["guide_identity"]
            .nunique(dropna=False)
        )
        conflicting_barcodes = conflicting[conflicting > 1].index.tolist()

        # Collapse exact duplicate annotation rows for the same barcode.
        # If a barcode has conflicting guide assignments, exclude its annotation
        # rather than arbitrarily choosing one guide.
        identity_dedup = identity_small.drop_duplicates(
            subset=["cell_barcode", "guide_identity"]
        )
        ambiguous_set = set(conflicting_barcodes)
        identity_dedup = identity_dedup[
            ~identity_dedup["cell_barcode"].isin(ambiguous_set)
        ]
        identity_dedup = identity_dedup.drop_duplicates(
            subset=["cell_barcode"], keep="first"
        )

        # Exact barcode join, never positional matching.
        aligned = barcode_map.merge(
            identity_dedup,
            on="cell_barcode",
            how="left",
            validate="one_to_one"
        )
        aligned.insert(0, "sample_id", sample)
        aligned["annotation_match"] = aligned["guide_identity"].notna()
        aligned["guide_class_provisional"] = aligned["guide_identity"].map(classify_guide)

        matched = int(aligned["annotation_match"].sum())
        unmatched_matrix = aligned.loc[~aligned["annotation_match"], "cell_barcode"].tolist()
        identity_barcodes = set(identity_small["cell_barcode"])
        matrix_barcodes = set(barcodes)
        unmatched_identity = sorted(identity_barcodes - matrix_barcodes)

        # All guide identities, not just top 100.
        guide_counts = (
            aligned.loc[aligned["annotation_match"]]
            .groupby(["guide_identity", "guide_class_provisional"], dropna=False)
            .size().reset_index(name="n_cells")
        )
        guide_counts.insert(0, "sample_id", sample)
        guide_counts["fraction_of_annotated_cells"] = (
            guide_counts["n_cells"] / max(matched, 1)
        )
        all_guide_counts.append(guide_counts)

        # Save mapping and per-sample discrepancy lists.
        aligned.to_csv(OUT / f"{sample}_cell_to_guide.csv.gz",
                       index=False, compression="gzip")
        pd.DataFrame({"unmatched_matrix_barcode": unmatched_matrix}).to_csv(
            OUT / f"{sample}_unmatched_matrix_barcodes.csv", index=False
        )
        pd.DataFrame({"unmatched_identity_barcode": unmatched_identity}).to_csv(
            OUT / f"{sample}_unmatched_identity_barcodes.csv", index=False
        )
        pd.DataFrame({"conflicting_identity_barcode": conflicting_barcodes}).to_csv(
            OUT / f"{sample}_conflicting_identity_barcodes.csv", index=False
        )
        guide_counts.to_csv(OUT / f"{sample}_guide_counts.csv", index=False)

        g = genes.copy()
        g.insert(0, "sample_id", sample)
        all_gene_annotations.append(g)

        report = {
            "sample_id": sample,
            "matrix_genes": int(matrix.shape[0]),
            "matrix_cells": int(matrix.shape[1]),
            "gene_annotation_rows": int(len(genes)),
            "barcode_rows": int(len(barcodes)),
            "identity_rows": int(len(identity)),
            "matched_cells": matched,
            "unmatched_matrix_barcodes": int(len(unmatched_matrix)),
            "unmatched_identity_barcodes": int(len(unmatched_identity)),
            "duplicate_matrix_barcodes": int(duplicate_matrix_barcodes),
            "duplicate_identity_barcodes": int(duplicate_identity_barcodes),
            "conflicting_guide_assignments": int(len(conflicting_barcodes)),
            "unique_guide_identities": int(aligned["guide_identity"].nunique()),
            "annotated_cell_fraction": float(matched / max(len(barcodes), 1)),
            "matrix_nnz": int(matrix.nnz),
            "matrix_orientation": "genes_by_cells",
        }
        sample_reports.append(report)

        print(f"Matrix cells:                  {len(barcodes):,}")
        print(f"Identity rows:                 {len(identity):,}")
        print(f"Matched by barcode:            {matched:,}")
        print(f"Unmatched matrix barcodes:     {len(unmatched_matrix):,}")
        print(f"Unmatched identity barcodes:   {len(unmatched_identity):,}")
        print(f"Duplicate matrix barcodes:     {duplicate_matrix_barcodes:,}")
        print(f"Duplicate identity barcodes:   {duplicate_identity_barcodes:,}")
        print(f"Conflicting guide assignments: {len(conflicting_barcodes):,}")
        print(f"Unique guide identities:       {aligned['guide_identity'].nunique():,}")
        print("Top 15 guide identities:")
        print(guide_counts.sort_values("n_cells", ascending=False)
              .head(15)[["guide_identity", "n_cells", "guide_class_provisional"]]
              .to_string(index=False))

        del matrix

    guide_all = pd.concat(all_guide_counts, ignore_index=True)
    guide_all.to_csv(OUT / "ALL_samples_guide_identity_counts.csv", index=False)

    # Combined mappings; sample_id keeps barcodes from different runs distinct.
    mapping_files = [OUT / f"{s}_cell_to_guide.csv.gz" for s in SAMPLES]
    mappings = [
        pd.read_csv(p, compression="gzip", low_memory=False)
        for p in mapping_files
    ]
    mapping_all = pd.concat(mappings, ignore_index=True)
    mapping_all.to_csv(OUT / "ALL_samples_cell_to_guide.csv.gz",
                       index=False, compression="gzip")

    gene_all = pd.concat(all_gene_annotations, ignore_index=True)
    gene_all.to_csv(OUT / "ALL_samples_gene_annotations.csv", index=False)

    report = {
        "samples": sample_reports,
        "total_matrix_cells": int(sum(x["matrix_cells"] for x in sample_reports)),
        "total_matched_cells": int(sum(x["matched_cells"] for x in sample_reports)),
        "total_unmatched_matrix_barcodes": int(
            sum(x["unmatched_matrix_barcodes"] for x in sample_reports)
        ),
        "total_unique_guide_strings_across_sample_rows": int(
            guide_all["guide_identity"].nunique()
        ),
        "outputs": {
            "guide_counts": str(OUT / "ALL_samples_guide_identity_counts.csv"),
            "cell_mapping": str(OUT / "ALL_samples_cell_to_guide.csv.gz"),
            "gene_annotations": str(OUT / "ALL_samples_gene_annotations.csv"),
        },
        "warning": (
            "Guide classes are provisional string-based categories. Confirm "
            "control identities and target-gene mappings from the original "
            "study documentation before using them as labels."
        )
    }
    with open(OUT / "alignment_and_guide_QC.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("BARCODE ALIGNMENT AND GUIDE INVENTORY COMPLETE")
    print("=" * 72)
    print(json.dumps(report, indent=2))
    print("\nSend alignment_and_guide_QC.json and ALL_samples_guide_identity_counts.csv.")

if __name__ == "__main__":
    main()

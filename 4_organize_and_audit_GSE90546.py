"""
Organize and audit GSE90546 supplementary files for the CRLP project.

Source:
    C:\\Users\\Naveen Joy\\Downloads\\GSE90546_RAW

Project:
    C:\\Users\\Naveen Joy\\Desktop\\BA

Actions:
    1. Discover sample files recursively.
    2. Copy all four file types for each sample, preserving .gz compression.
    3. Validate sample completeness.
    4. Read Matrix Market files as sparse matrices (never densify).
    5. Check gene/barcode dimensions and inspect cell-identity annotations.
    6. Write CSV/JSON QC reports and small metadata previews.

This script does NOT create compensation labels. It audits the raw evidence
needed to determine whether perturbation-response labels can be generated.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.io import mmread
from scipy import sparse

# ----------------------------- PATHS ---------------------------------

SOURCE_DIR = Path(r"C:\Users\Naveen Joy\Downloads\GSE90546_RAW")
PROJECT_DIR = Path(__file__).resolve().parent
DEST_DIR = PROJECT_DIR / "datasets" / "GSE90546_RAW"
QC_DIR = PROJECT_DIR / "preprocessed" / "GSE90546_QC"

# GSE file suffixes and their semantic roles.
SUFFIXES = {
    "_barcodes.tsv.gz": "barcodes",
    "_cell_identities.csv.gz": "cell_identities",
    "_genes.tsv.gz": "genes",
    "_matrix.mtx.txt.gz": "matrix",
}

# ----------------------------- HELPERS -------------------------------

def stop(message: str, code: int = 1):
    print(f"\nERROR: {message}")
    raise SystemExit(code)


def identify_file(path: Path):
    name = path.name
    for suffix, role in SUFFIXES.items():
        if name.endswith(suffix):
            sample_id = name[:-len(suffix)]
            return sample_id, role
    return None, None


def read_first_lines_gz(path: Path, n=5):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for _ in range(n):
            line = handle.readline()
            if not line:
                break
            rows.append(line.rstrip("\r\n"))
    return rows


def count_tsv_rows(path: Path):
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            count += 1
    return count


def inspect_cell_identity(path: Path):
    # These files are expected to be small; read the full annotation table.
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    result = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "preview": df.head(5).fillna("").astype(str).to_dict(orient="records"),
        "unique_counts": {},
        "candidate_annotation_columns": [],
    }

    # Summarise low-cardinality columns, useful for identifying perturbation labels.
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        if nunique <= 100:
            vals = df[col].dropna().astype(str).value_counts().head(30)
            result["unique_counts"][str(col)] = {
                str(k): int(v) for k, v in vals.items()
            }

        name = str(col).lower()
        if any(term in name for term in (
            "guide", "grna", "perturb", "target", "gene", "identity",
            "condition", "treatment", "control", "sample", "crispr",
            "knock", "cell"
        )):
            result["candidate_annotation_columns"].append(str(col))

    return df, result


# ----------------------------- PRECHECK -------------------------------

print("=" * 78)
print("GSE90546 ORGANIZATION AND QC")
print("=" * 78)
print(f"Source:  {SOURCE_DIR}")
print(f"Project: {PROJECT_DIR}")
print(f"Target:  {DEST_DIR}")

if not SOURCE_DIR.exists():
    stop(
        "The extracted source folder was not found. Confirm that "
        r"C:\Users\Naveen Joy\Downloads\GSE90546_RAW exists."
    )

if not PROJECT_DIR.exists():
    stop(f"Project directory does not exist: {PROJECT_DIR}")

DEST_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

all_files = [p for p in SOURCE_DIR.rglob("*") if p.is_file()]
sample_files = {}

for path in all_files:
    sample_id, role = identify_file(path)
    if sample_id is not None:
        sample_files.setdefault(sample_id, {})[role] = path

if not sample_files:
    stop(
        "No expected .gz files were found. Expected filenames ending in "
        "_barcodes.tsv.gz, _cell_identities.csv.gz, _genes.tsv.gz, "
        "or _matrix.mtx.txt.gz."
    )

print(f"\nDiscovered {len(sample_files)} sample groups.")
print(f"Discovered {sum(len(v) for v in sample_files.values())} matching files.")

# ----------------------------- COPY FILES -----------------------------

copy_records = []
missing_records = []

for sample_id, role_map in sorted(sample_files.items()):
    print(f"\nSample: {sample_id}")

    for role, suffix in [
        ("barcodes", "_barcodes.tsv.gz"),
        ("cell_identities", "_cell_identities.csv.gz"),
        ("genes", "_genes.tsv.gz"),
        ("matrix", "_matrix.mtx.txt.gz"),
    ]:
        src = role_map.get(role)

        if src is None:
            missing_records.append({
                "sample_id": sample_id,
                "missing_role": role,
                "expected_suffix": suffix,
            })
            print(f"  MISSING: {role}")
            continue

        dst = DEST_DIR / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

        copy_records.append({
            "sample_id": sample_id,
            "role": role,
            "filename": dst.name,
            "source_path": str(src),
            "destination_path": str(dst),
            "size_bytes": int(dst.stat().st_size),
        })
        print(f"  copied {role}: {dst.name} ({dst.stat().st_size:,} bytes)")

pd.DataFrame(copy_records).to_csv(QC_DIR / "copied_files.csv", index=False)
pd.DataFrame(missing_records).to_csv(QC_DIR / "missing_files.csv", index=False)

# ----------------------------- SAMPLE QC ------------------------------

sample_qc = []
identity_summaries = {}
identity_tables = {}

for sample_id, role_map in sorted(sample_files.items()):
    print(f"\nAuditing sample: {sample_id}")
    row = {
        "sample_id": sample_id,
        "has_barcodes": "barcodes" in role_map,
        "has_cell_identities": "cell_identities" in role_map,
        "has_genes": "genes" in role_map,
        "has_matrix": "matrix" in role_map,
        "complete_four_file_set": all(
            k in role_map for k in ("barcodes", "cell_identities", "genes", "matrix")
        ),
    }

    # Use copied files in the project folder.
    paths = {
        role: DEST_DIR / path.name
        for role, path in role_map.items()
    }

    gene_count = None
    barcode_count = None
    matrix_shape = None
    matrix_nnz = None
    matrix_orientation = None

    if "genes" in paths:
        gene_count = count_tsv_rows(paths["genes"])
        row["gene_rows"] = gene_count
        row["genes_preview"] = read_first_lines_gz(paths["genes"], 3)

    if "barcodes" in paths:
        barcode_count = count_tsv_rows(paths["barcodes"])
        row["barcode_rows"] = barcode_count
        row["barcodes_preview"] = read_first_lines_gz(paths["barcodes"], 3)

    if "matrix" in paths:
        print("  Reading sparse Matrix Market file...")
        # mmread accepts a text stream; the matrix remains sparse.
        with gzip.open(paths["matrix"], "rt", encoding="ascii") as handle:
            matrix = mmread(handle)

        if not sparse.issparse(matrix):
            matrix = sparse.coo_matrix(matrix)
        matrix = matrix.tocsr()

        matrix_shape = tuple(int(x) for x in matrix.shape)
        matrix_nnz = int(matrix.nnz)
        row["matrix_rows"] = matrix_shape[0]
        row["matrix_columns"] = matrix_shape[1]
        row["matrix_nonzero_entries"] = matrix_nnz
        row["matrix_dtype"] = str(matrix.dtype)

        # 10x-style matrices are commonly genes x cells, but verify dimensions.
        if gene_count is not None and barcode_count is not None:
            if matrix_shape == (gene_count, barcode_count):
                matrix_orientation = "genes_by_cells"
            elif matrix_shape == (barcode_count, gene_count):
                matrix_orientation = "cells_by_genes"
            else:
                matrix_orientation = "dimension_mismatch"
        row["matrix_orientation"] = matrix_orientation

        print(
            f"  Matrix shape={matrix_shape}, nnz={matrix_nnz:,}, "
            f"orientation={matrix_orientation}"
        )

        del matrix

    if "cell_identities" in paths:
        df, summary = inspect_cell_identity(paths["cell_identities"])
        identity_tables[sample_id] = df
        identity_summaries[sample_id] = summary
        row["identity_rows"] = int(df.shape[0])
        row["identity_columns"] = int(df.shape[1])
        row["identity_column_names"] = " | ".join(map(str, df.columns))
        row["candidate_annotation_columns"] = " | ".join(
            summary["candidate_annotation_columns"]
        )

        # Keep a per-sample preview for quick inspection.
        df.head(100).to_csv(
            QC_DIR / f"{sample_id}_cell_identity_preview.csv", index=False
        )

        # Compare annotation rows to barcode count where possible.
        if barcode_count is not None:
            row["identity_rows_match_barcodes"] = (
                int(df.shape[0]) == int(barcode_count)
            )

    sample_qc.append(row)

    if not row["complete_four_file_set"]:
        print("  WARNING: incomplete four-file sample set.")
    if matrix_orientation == "dimension_mismatch":
        print("  WARNING: matrix dimensions do not match gene/barcode counts.")

# ----------------------------- COMBINED IDENTITY AUDIT ----------------

# Save a combined metadata table with a sample_id column.
combined = []
for sample_id, df in identity_tables.items():
    copy_df = df.copy()
    copy_df.insert(0, "source_sample_id", sample_id)
    combined.append(copy_df)

if combined:
    combined_df = pd.concat(combined, ignore_index=True, sort=False)
    combined_df.to_csv(QC_DIR / "combined_cell_identities.csv.gz",
                       index=False, compression="gzip")

    # Summaries for likely perturbation/control columns.
    candidate_cols = set()
    for summary in identity_summaries.values():
        candidate_cols.update(summary["candidate_annotation_columns"])

    annotation_report = {
        "combined_rows": int(len(combined_df)),
        "combined_columns": [str(c) for c in combined_df.columns],
        "candidate_annotation_columns": sorted(candidate_cols),
        "column_value_counts": {},
    }

    for col in sorted(candidate_cols):
        if col in combined_df.columns:
            counts = combined_df[col].dropna().astype(str).value_counts().head(100)
            annotation_report["column_value_counts"][col] = {
                str(k): int(v) for k, v in counts.items()
            }

    with open(QC_DIR / "annotation_summary.json", "w", encoding="utf-8") as f:
        json.dump(annotation_report, f, indent=2, default=str)

# ----------------------------- FINAL REPORT ---------------------------

sample_qc_df = pd.DataFrame(sample_qc)
sample_qc_df.to_csv(QC_DIR / "sample_matrix_QC.csv", index=False)

report = {
    "source_directory": str(SOURCE_DIR),
    "project_directory": str(PROJECT_DIR),
    "destination_directory": str(DEST_DIR),
    "qc_directory": str(QC_DIR),
    "sample_count": int(len(sample_files)),
    "matching_file_count": int(sum(len(v) for v in sample_files.values())),
    "complete_sample_count": int(
        sample_qc_df["complete_four_file_set"].sum()
    ),
    "incomplete_sample_count": int(
        (~sample_qc_df["complete_four_file_set"]).sum()
    ),
    "missing_file_records": missing_records,
    "sample_qc": sample_qc,
    "identity_summaries": identity_summaries,
    "interpretation": (
        "This stage organizes and audits raw perturbation data. It does not "
        "generate functional compensation labels. Those require independent "
        "functional evidence beyond expression changes."
    ),
}

with open(QC_DIR / "GSE90546_import_QC.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)

print("\n" + "=" * 78)
print("IMPORT AND QC FINISHED")
print("=" * 78)
print(f"Samples discovered: {len(sample_files)}")
print(f"Complete samples:   {report['complete_sample_count']}")
print(f"Incomplete samples: {report['incomplete_sample_count']}")
print(f"Project data folder: {DEST_DIR}")
print(f"QC reports folder:   {QC_DIR}")
print("\nImportant output files:")
print(f"  {QC_DIR / 'sample_matrix_QC.csv'}")
print(f"  {QC_DIR / 'annotation_summary.json'}")
print(f"  {QC_DIR / 'GSE90546_import_QC.json'}")
print(f"  {QC_DIR / 'combined_cell_identities.csv.gz'}")
print("\nNext: inspect annotation_summary.json and sample_matrix_QC.csv.")
print("Do not train a compensation classifier until labels are defensible.")

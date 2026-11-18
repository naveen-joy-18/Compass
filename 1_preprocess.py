
"""
CRLP - DepMap preprocessing pipeline
------------------------------------
1. Load Model metadata.
2. Identify common cell lines across CRISPR and expression datasets.
3. Identify overlapping gene columns.
4. Filter and export aligned matrices in chunks.
5. Generate QC reports.

Original datasets remain unchanged.
"""

import os
import gzip
import json
import pandas as pd
import numpy as np

from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
OUT_DIR = os.path.join(BASE_DIR, "preprocessed")

os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "effect": "CRISPRGeneEffect.csv",
    "expression": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "model": "Model.csv"
}

CHUNK_SIZE = 100

# ============================================================
# HELPERS
# ============================================================

def get_path(filename):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    return path


def inspect_header(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()

    # First column contains the cell-line/model identifier.
    id_column = header[0]

    # Remaining columns represent genes.
    gene_columns = header[1:]

    return id_column, gene_columns


def read_model_metadata(path):

    print("\n[1/5] Loading Model metadata...")

    model = pd.read_csv(path, low_memory=False)

    if "ModelID" not in model.columns:
        raise ValueError("Model.csv does not contain ModelID")

    model["ModelID"] = model["ModelID"].astype("string").str.strip()

    model = model.dropna(subset=["ModelID"])
    model = model.drop_duplicates(subset=["ModelID"])

    print("Metadata rows:", len(model))
    print("Metadata columns:", len(model.columns))

    return model


def read_matrix_ids(path):

    # Read only the first column, not the full gene matrix.
    ids = pd.read_csv(
        path,
        usecols=[0],
        dtype="string"
    ).iloc[:, 0]

    ids = ids.str.strip()
    ids = ids.dropna()

    return set(ids)


def export_filtered_matrix(
    path,
    output_path,
    id_column,
    gene_columns,
    selected_models
):

    print(f"\nProcessing: {os.path.basename(path)}")

    use_columns = [id_column] + gene_columns

    total_written = 0
    first_chunk = True
    missing_values = 0
    total_values = 0

    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as out:

        reader = pd.read_csv(
            path,
            usecols=use_columns,
            chunksize=CHUNK_SIZE,
            dtype={id_column: "string"},
            low_memory=False
        )

        for chunk in tqdm(reader, desc="Filtering rows"):

            chunk[id_column] = chunk[id_column].str.strip()

            # Keep only cell lines present in all datasets.
            chunk = chunk[
                chunk[id_column].isin(selected_models)
            ]

            if chunk.empty:
                continue

            # Convert gene measurements to numeric.
            for gene in gene_columns:
                chunk[gene] = pd.to_numeric(
                    chunk[gene],
                    errors="coerce"
                )

            missing_values += int(chunk[gene_columns].isna().sum().sum())
            total_values += len(chunk) * len(gene_columns)

            chunk = chunk.rename(columns={id_column: "ModelID"})

            chunk.to_csv(
                out,
                index=False,
                header=first_chunk
            )

            first_chunk = False
            total_written += len(chunk)

    return {
        "output": output_path,
        "rows_written": total_written,
        "gene_count": len(gene_columns),
        "missing_values": missing_values,
        "total_values": total_values
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("CRLP - DepMap PREPROCESSING")
    print("=" * 65)

    effect_path = get_path(FILES["effect"])
    expression_path = get_path(FILES["expression"])
    model_path = get_path(FILES["model"])

    # --------------------------------------------------------
    # STEP 1: Metadata
    # --------------------------------------------------------

    model = read_model_metadata(model_path)

    model_ids = set(model["ModelID"])

    # --------------------------------------------------------
    # STEP 2: Inspect gene columns
    # --------------------------------------------------------

    print("\n[2/5] Inspecting matrix headers...")

    effect_id, effect_genes = inspect_header(effect_path)

    expression_id, expression_genes = inspect_header(
        expression_path
    )

    print("CRISPR gene columns:", len(effect_genes))
    print("Expression gene columns:", len(expression_genes))

    # Exact matching preserves the gene symbol and identifier.
    common_genes = sorted(
        set(effect_genes).intersection(expression_genes)
    )

    print("Common genes:", len(common_genes))

    if not common_genes:
        raise ValueError("No overlapping gene columns found.")

    # --------------------------------------------------------
    # STEP 3: Identify common cell lines
    # --------------------------------------------------------

    print("\n[3/5] Reading cell-line identifiers...")

    effect_ids = read_matrix_ids(effect_path)

    expression_ids = read_matrix_ids(expression_path)

    common_models = (
        model_ids
        .intersection(effect_ids)
        .intersection(expression_ids)
    )

    common_models = sorted(common_models)

    print("Models in metadata:", len(model_ids))
    print("Models in CRISPR:", len(effect_ids))
    print("Models in expression:", len(expression_ids))
    print("Common models:", len(common_models))

    if not common_models:
        raise ValueError("No common cell lines found.")

    # --------------------------------------------------------
    # STEP 4: Save matched metadata
    # --------------------------------------------------------

    print("\n[4/5] Saving matched metadata...")

    matched_model = model[
        model["ModelID"].isin(common_models)
    ].copy()

    metadata_path = os.path.join(
        OUT_DIR,
        "Model_metadata_matched.csv"
    )

    matched_model.to_csv(metadata_path, index=False)

    # Save the exact gene-column list.
    genes_path = os.path.join(
        OUT_DIR,
        "common_genes.txt"
    )

    with open(genes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(common_genes))

    # --------------------------------------------------------
    # STEP 5: Export aligned matrices
    # --------------------------------------------------------

    print("\n[5/5] Exporting aligned matrices...")

    effect_output = os.path.join(
        OUT_DIR,
        "CRISPRGeneEffect_matched.csv.gz"
    )

    expression_output = os.path.join(
        OUT_DIR,
        "Expression_matched.csv.gz"
    )

    effect_report = export_filtered_matrix(
        effect_path,
        effect_output,
        effect_id,
        common_genes,
        set(common_models)
    )

    expression_report = export_filtered_matrix(
        expression_path,
        expression_output,
        expression_id,
        common_genes,
        set(common_models)
    )

    # --------------------------------------------------------
    # QC REPORT
    # --------------------------------------------------------

    report = {
        "project": "CRLP",
        "input_files": FILES,
        "metadata_models": len(model_ids),
        "common_models": len(common_models),
        "effect_gene_columns": len(effect_genes),
        "expression_gene_columns": len(expression_genes),
        "common_genes": len(common_genes),
        "effect": effect_report,
        "expression": expression_report,
        "outputs": {
            "metadata": metadata_path,
            "effect": effect_output,
            "expression": expression_output,
            "genes": genes_path
        }
    }

    report_path = os.path.join(
        OUT_DIR,
        "preprocessing_QC.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETED")
    print("=" * 65)

    print("Common models:", len(common_models))
    print("Common genes:", len(common_genes))

    print("\nMissing-value percentages:")

    for name, result in [
        ("CRISPR effect", effect_report),
        ("Expression", expression_report)
    ]:
        percentage = (
            100 * result["missing_values"] / result["total_values"]
            if result["total_values"] else 0
        )

        print(f"{name}: {percentage:.3f}%")

    print("\nOutput directory:", OUT_DIR)
    print("QC report:", report_path)


if __name__ == "__main__":
    main()


"""
CRLP - Feature Engineering Stage 1
===================================

Input:
    CRISPRGeneEffect_matched.csv.gz
    Expression_matched.csv.gz
    Model_metadata_matched.csv

Outputs:
    gene_level_features.csv
    candidate_gene_pairs.csv.gz
    feature_engineering_QC.json

Features:
    Baseline expression statistics
    CRISPR dependency statistics
    Expression-profile correlation
    Dependency-profile correlation
    Missingness

IMPORTANT:
    These are candidate-generation and supporting features.
    They are NOT compensation labels.
"""

import os
import json
import numpy as np
import pandas as pd

from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "preprocessed")
OUT_DIR = os.path.join(BASE_DIR, "features")

os.makedirs(OUT_DIR, exist_ok=True)

EFFECT_FILE = "CRISPRGeneEffect_matched.csv.gz"
EXPRESSION_FILE = "Expression_matched.csv.gz"
MODEL_FILE = "Model_metadata_matched.csv"

TOP_K = 10
CHUNK_SIZE = 128

# ============================================================
# HELPERS
# ============================================================

def load_matrix(filename):

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"\nLoading {filename}")

    df = pd.read_csv(path, compression="gzip")

    if "ModelID" not in df.columns:
        raise ValueError(f"ModelID missing in {filename}")

    if df["ModelID"].duplicated().any():
        raise ValueError(f"Duplicate ModelIDs in {filename}")

    df["ModelID"] = df["ModelID"].astype(str)

    return df


def make_numeric_matrix(df, genes):

    return df[genes].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=np.float32)


def normalize_profiles(matrix):

    """
    Normalize gene profiles across cell lines.

    Missing values are replaced by the corresponding
    gene's mean ONLY for correlation calculation.

    Original missingness is retained separately.
    """

    means = np.nanmean(matrix, axis=0)

    means = np.where(
        np.isfinite(means),
        means,
        0.0
    )

    filled = np.where(
        np.isfinite(matrix),
        matrix,
        means[None, :]
    ).astype(np.float32)

    centered = filled - means[None, :]

    norms = np.sqrt(
        np.sum(centered ** 2, axis=0)
    )

    valid = norms > 1e-8

    normalized = np.zeros_like(centered)

    normalized[:, valid] = (
        centered[:, valid] / norms[None, valid]
    )

    return normalized, valid


def find_top_correlations(
    normalized,
    valid,
    gene_names,
    top_k,
    chunk_size,
    description
):

    """
    Find top absolute profile correlations per target.

    Calculates one target block against all genes.
    Does not construct a complete square matrix.
    """

    n_genes = len(gene_names)

    neighbors = {}

    for start in tqdm(
        range(0, n_genes, chunk_size),
        desc=description
    ):

        end = min(start + chunk_size, n_genes)

        correlations = (
            normalized[:, start:end].T @ normalized
        )

        for local_idx in range(end - start):

            target_idx = start + local_idx

            if not valid[target_idx]:
                continue

            row = correlations[local_idx].copy()

            row[target_idx] = np.nan

            row[~valid] = np.nan

            finite = np.isfinite(row)

            if finite.sum() == 0:
                continue

            candidate_indices = np.flatnonzero(finite)

            k = min(top_k, len(candidate_indices))

            selected = candidate_indices[
                np.argpartition(
                    np.abs(row[candidate_indices]),
                    -k
                )[-k:]
            ]

            selected = selected[
                np.argsort(
                    -np.abs(row[selected])
                )
            ]

            neighbors[target_idx] = [
                (int(idx), float(row[idx]))
                for idx in selected
            ]

    return neighbors


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("CRLP - FEATURE ENGINEERING")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    effect_df = load_matrix(EFFECT_FILE)
    expression_df = load_matrix(EXPRESSION_FILE)

    metadata = pd.read_csv(
        os.path.join(DATA_DIR, MODEL_FILE)
    )

    # --------------------------------------------------------
    # 2. ALIGN MODELS AND GENES
    # --------------------------------------------------------

    print("\n[1/6] Aligning datasets...")

    effect_df["ModelID"] = effect_df["ModelID"].astype(str)
    expression_df["ModelID"] = expression_df["ModelID"].astype(str)

    common_models = sorted(
        set(effect_df["ModelID"])
        .intersection(expression_df["ModelID"])
    )

    effect_df = (
        effect_df.set_index("ModelID")
        .loc[common_models]
    )

    expression_df = (
        expression_df.set_index("ModelID")
        .loc[common_models]
    )

    common_genes = sorted(
        set(effect_df.columns)
        .intersection(expression_df.columns)
    )

    print("Common models:", len(common_models))
    print("Common genes:", len(common_genes))

    if not common_models or not common_genes:
        raise ValueError("No common models or genes.")

    # --------------------------------------------------------
    # 3. NUMERIC MATRICES
    # --------------------------------------------------------

    print("\n[2/6] Creating numeric matrices...")

    effect = make_numeric_matrix(
        effect_df,
        common_genes
    )

    expression = make_numeric_matrix(
        expression_df,
        common_genes
    )

    print("Effect matrix:", effect.shape)
    print("Expression matrix:", expression.shape)

    # --------------------------------------------------------
    # 4. GENE-LEVEL FEATURES
    # --------------------------------------------------------

    print("\n[3/6] Calculating gene-level features...")

    gene_features = pd.DataFrame({
        "gene": common_genes,

        "expression_mean": np.nanmean(
            expression, axis=0
        ),

        "expression_std": np.nanstd(
            expression, axis=0
        ),

        "expression_missing_fraction": np.mean(
            ~np.isfinite(expression), axis=0
        ),

        "dependency_mean": np.nanmean(
            effect, axis=0
        ),

        "dependency_std": np.nanstd(
            effect, axis=0
        ),

        "dependency_missing_fraction": np.mean(
            ~np.isfinite(effect), axis=0
        )
    })

    gene_features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    gene_feature_path = os.path.join(
        OUT_DIR,
        "gene_level_features.csv"
    )

    gene_features.to_csv(
        gene_feature_path,
        index=False
    )

    # --------------------------------------------------------
    # 5. PROFILE NORMALIZATION
    # --------------------------------------------------------

    print("\n[4/6] Normalizing gene profiles...")

    expression_norm, expression_valid = (
        normalize_profiles(expression)
    )

    effect_norm, effect_valid = (
        normalize_profiles(effect)
    )

    # --------------------------------------------------------
    # 6. TOP CORRELATED CANDIDATES
    # --------------------------------------------------------

    print("\n[5/6] Finding expression-profile neighbors...")

    expression_neighbors = find_top_correlations(
        expression_norm,
        expression_valid,
        common_genes,
        TOP_K,
        CHUNK_SIZE,
        "Expression correlations"
    )

    print("\nFinding dependency-profile neighbors...")

    dependency_neighbors = find_top_correlations(
        effect_norm,
        effect_valid,
        common_genes,
        TOP_K,
        CHUNK_SIZE,
        "Dependency correlations"
    )

    # --------------------------------------------------------
    # 7. BUILD PAIR-LEVEL FEATURES
    # --------------------------------------------------------

    print("\n[6/6] Constructing pair-level features...")

    gene_to_index = {
        gene: idx
        for idx, gene in enumerate(common_genes)
    }

    gene_lookup = gene_features.set_index("gene")

    pairs = {}

    for target_idx, candidate_list in expression_neighbors.items():

        target = common_genes[target_idx]

        for candidate_idx, corr in candidate_list:

            candidate = common_genes[candidate_idx]

            pairs.setdefault(
                (target, candidate),
                {}
            )["expression_correlation"] = corr

    for target_idx, candidate_list in dependency_neighbors.items():

        target = common_genes[target_idx]

        for candidate_idx, corr in candidate_list:

            candidate = common_genes[candidate_idx]

            pairs.setdefault(
                (target, candidate),
                {}
            )["dependency_correlation"] = corr

    rows = []

    for (target, candidate), correlations in tqdm(
        pairs.items(),
        desc="Building gene-pair features"
    ):

        target_features = gene_lookup.loc[target]
        candidate_features = gene_lookup.loc[candidate]

        rows.append({

            "target_gene": target,
            "candidate_gene": candidate,

            "target_expression_mean":
                target_features["expression_mean"],

            "candidate_expression_mean":
                candidate_features["expression_mean"],

            "target_expression_std":
                target_features["expression_std"],

            "candidate_expression_std":
                candidate_features["expression_std"],

            "target_dependency_mean":
                target_features["dependency_mean"],

            "candidate_dependency_mean":
                candidate_features["dependency_mean"],

            "target_dependency_std":
                target_features["dependency_std"],

            "candidate_dependency_std":
                candidate_features["dependency_std"],

            "target_dependency_missing_fraction":
                target_features["dependency_missing_fraction"],

            "candidate_dependency_missing_fraction":
                candidate_features["dependency_missing_fraction"],

            "expression_correlation":
                correlations.get(
                    "expression_correlation",
                    np.nan
                ),

            "dependency_correlation":
                correlations.get(
                    "dependency_correlation",
                    np.nan
                ),

            "expression_mean_difference":
                abs(
                    target_features["expression_mean"]
                    - candidate_features["expression_mean"]
                ),

            "dependency_mean_difference":
                abs(
                    target_features["dependency_mean"]
                    - candidate_features["dependency_mean"]
                ),

            "candidate_source":
                "+".join(sorted(correlations.keys()))
        })

    pair_df = pd.DataFrame(rows)

    pair_output = os.path.join(
        OUT_DIR,
        "candidate_gene_pairs.csv.gz"
    )

    pair_df.to_csv(
        pair_output,
        index=False,
        compression="gzip"
    )

    # --------------------------------------------------------
    # 8. QC REPORT
    # --------------------------------------------------------

    report = {

        "models": len(common_models),

        "genes": len(common_genes),

        "gene_level_features": len(gene_features.columns) - 1,

        "candidate_pairs": len(pair_df),

        "top_k": TOP_K,

        "expression_neighbor_targets":
            len(expression_neighbors),

        "dependency_neighbor_targets":
            len(dependency_neighbors),

        "pair_features": list(pair_df.columns),

        "correlation_missing_value_handling":
            "Per-gene mean imputation for correlation calculation only",

        "compensation_labels_created": False,

        "interpretation":
            "Candidate pairs are statistical neighbors, not validated compensatory relationships."
    }

    qc_path = os.path.join(
        OUT_DIR,
        "feature_engineering_QC.json"
    )

    with open(qc_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 65)

    print("Models:", len(common_models))
    print("Genes:", len(common_genes))
    print("Candidate pairs:", len(pair_df))

    print("\nOutput directory:", OUT_DIR)

    print("\nGenerated files:")
    print(gene_feature_path)
    print(pair_output)
    print(qc_path)


if __name__ == "__main__":
    main()

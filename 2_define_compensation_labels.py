
"""
CRLP: Evidence-based compensatory relationship labelling

Labels:
    1 = Positive
    0 = Negative
   -1 = Uncertain

Important:
- No labels are inferred from DepMap expression or gene-effect scores.
- Positive labels require induced response AND functional compensation evidence.
- Negative labels require an adequate experiment demonstrating no
  compensatory response.
- All other cases remain uncertain.
"""

import os
import pandas as pd
import numpy as np
import json

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed")

TEMPLATE_FILE = os.path.join(
    OUTPUT_DIR,
    "compensation_evidence_template.csv"
)

ANNOTATED_FILE = os.path.join(
    OUTPUT_DIR,
    "compensation_evidence_annotated.csv"
)

LABELLED_FILE = os.path.join(
    OUTPUT_DIR,
    "compensation_labels_all.csv"
)

TRAINING_FILE = os.path.join(
    OUTPUT_DIR,
    "compensation_training_pairs.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "compensation_label_QC.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# REQUIRED ANNOTATION COLUMNS
# ============================================================

COLUMNS = [
    "target_gene",
    "candidate_gene",

    # Experimental context
    "cell_type",
    "perturbation_method",
    "timepoint",

    # Biological evidence
    "induced_response",
    "functional_relationship",
    "functional_compensation_or_rescue",

    # Evidence required for negative classification
    "adequate_negative_test",
    "observed_no_compensation",

    # Supporting documentation
    "evidence_reference",
    "evidence_description",
    "curator_notes"
]

BOOLEAN_COLUMNS = [
    "induced_response",
    "functional_relationship",
    "functional_compensation_or_rescue",
    "adequate_negative_test",
    "observed_no_compensation"
]

# ============================================================
# STEP 1: CREATE ANNOTATION TEMPLATE
# ============================================================

def create_template():

    if os.path.exists(TEMPLATE_FILE):
        print("Annotation template already exists.")
        return

    template = pd.DataFrame(columns=COLUMNS)

    template.to_csv(TEMPLATE_FILE, index=False)

    print("\nCreated annotation template:")
    print(TEMPLATE_FILE)


# ============================================================
# STEP 2: PARSE BOOLEAN VALUES
# ============================================================

def parse_boolean(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in ["true", "yes", "1"]:
        return True

    if value in ["false", "no", "0"]:
        return False

    if value in ["", "nan", "none", "unknown", "na"]:
        return None

    raise ValueError(
        f"Invalid boolean value: {value}. "
        "Use TRUE, FALSE or UNKNOWN."
    )


# ============================================================
# STEP 3: CLASSIFY INDIVIDUAL EVIDENCE RECORDS
# ============================================================

def classify_record(row):

    induced = row["induced_response"]
    relationship = row["functional_relationship"]
    rescue = row["functional_compensation_or_rescue"]

    adequate = row["adequate_negative_test"]
    no_compensation = row["observed_no_compensation"]

    reference = str(row["evidence_reference"]).strip()

    has_reference = (
        reference not in ["", "nan", "None"]
    )

    # Positive evidence:
    # An induced response, a plausible functional relationship,
    # and direct evidence of functional compensation/rescue.

    positive = (
        induced is True
        and relationship is True
        and rescue is True
        and has_reference
    )

    # Negative evidence:
    # A sufficiently informative experiment explicitly shows
    # no compensatory response.

    negative = (
        adequate is True
        and no_compensation is True
        and induced is False
        and rescue is False
        and has_reference
    )

    if positive and negative:
        return "uncertain", -1, "Conflicting evidence"

    if positive:
        return "positive", 1, "Supported compensation"

    if negative:
        return "negative", 0, "Adequately tested; no compensation observed"

    return "uncertain", -1, "Insufficient or incomplete evidence"


# ============================================================
# STEP 4: VALIDATE INPUT
# ============================================================

def validate_input(df):

    missing = set(COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[COLUMNS].copy()

    # Remove completely empty records.
    df = df.dropna(
        how="all",
        subset=["target_gene", "candidate_gene"]
    )

    df["target_gene"] = (
        df["target_gene"].astype("string").str.strip().str.upper().str.replace(r"\s*\(\d+\)$", "", regex=True)
    )

    df["candidate_gene"] = (
        df["candidate_gene"].astype("string").str.strip().str.upper().str.replace(r"\s*\(\d+\)$", "", regex=True)
    )

    # Self-pairs are invalid.
    self_pairs = (
        df["target_gene"] == df["candidate_gene"]
    )

    if self_pairs.any():
        print(
            "Removing self-pairs:",
            int(self_pairs.sum())
        )

        df = df.loc[~self_pairs].copy()

    # Reject missing gene names.
    df = df.dropna(
        subset=["target_gene", "candidate_gene"]
    )

    # Parse evidence fields.
    for column in BOOLEAN_COLUMNS:

        df[column] = df[column].apply(parse_boolean)

    return df


# ============================================================
# STEP 5: LABEL DATA
# ============================================================

def generate_labels():

    if not os.path.exists(ANNOTATED_FILE):

        print("\nAnnotated evidence file not found.")
        print("First fill the template and save it as:")
        print(ANNOTATED_FILE)

        return

    print("\nReading annotated evidence...")

    df = pd.read_csv(
        ANNOTATED_FILE,
        dtype=str
    )

    df = validate_input(df)

    if df.empty:
        print("No annotated records found.")
        return

    print("Evidence records:", len(df))

    # Classify each experimental-context record.
    results = df.apply(
        classify_record,
        axis=1,
        result_type="expand"
    )

    results.columns = [
        "label",
        "label_numeric",
        "classification_reason"
    ]

    df = pd.concat(
        [df.reset_index(drop=True),
         results.reset_index(drop=True)],
        axis=1
    )

    # ========================================================
    # STEP 6: CHECK DUPLICATE CONTEXT RECORDS
    # ========================================================

    context_columns = [
        "target_gene",
        "candidate_gene",
        "cell_type",
        "perturbation_method",
        "timepoint"
    ]

    duplicated = df.duplicated(
        subset=context_columns,
        keep=False
    )

    if duplicated.any():

        print(
            "\nWARNING:",
            int(duplicated.sum()),
            "records share the same experimental context."
        )

        print(
            "Review these records for duplicate or conflicting evidence."
        )

    # ========================================================
    # STEP 7: EXPORT LABELLED DATA
    # ========================================================

    df.to_csv(
        LABELLED_FILE,
        index=False
    )

    # Only positive and negative records enter supervised training.
    training = df[
        df["label"].isin(["positive", "negative"])
    ].copy()

    training.to_csv(
        TRAINING_FILE,
        index=False
    )

    # ========================================================
    # STEP 8: QC REPORT
    # ========================================================

    counts = (
        df["label"]
        .value_counts()
        .to_dict()
    )

    report = {
        "total_evidence_records": int(len(df)),
        "positive_records": int(counts.get("positive", 0)),
        "negative_records": int(counts.get("negative", 0)),
        "uncertain_records": int(counts.get("uncertain", 0)),
        "training_records": int(len(training)),
        "unique_target_genes": int(df["target_gene"].nunique()),
        "unique_candidate_genes": int(df["candidate_gene"].nunique()),
        "unique_gene_pairs": int(
            df[["target_gene", "candidate_gene"]]
            .drop_duplicates()
            .shape[0]
        ),
        "training_positive": int(
            (training["label"] == "positive").sum()
        ),
        "training_negative": int(
            (training["label"] == "negative").sum()
        )
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("COMPENSATION LABEL GENERATION COMPLETED")
    print("=" * 60)

    print("Total evidence records:", len(df))
    print("Positive:", counts.get("positive", 0))
    print("Negative:", counts.get("negative", 0))
    print("Uncertain:", counts.get("uncertain", 0))

    print("\nTraining records:", len(training))

    print("\nOutput files:")
    print(LABELLED_FILE)
    print(TRAINING_FILE)
    print(REPORT_FILE)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_template()

    generate_labels()

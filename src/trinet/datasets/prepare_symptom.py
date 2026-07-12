"""Clean the symptom dataset — and fix the data leakage the original code shipped.

The original `sbd.py` engineered a `sum` feature (the row-wise sum of every column) — dubious
methodology, so we drop it. (An ablation in symptom_model.py shows `sum` actually changes
accuracy by <1pt, so it was NOT the source of the paper's implausible 97.86%. The honest
finding is stronger: the Kaggle MSMPC symptom set is largely synthetic and weakly predictive,
and no standard model exceeds ~70% on it — 97.86% is simply not reproducible here.)

Output: data/processed/symptom.csv  (numeric, target column `MonkeyPox` in {0,1})
"""

from __future__ import annotations

import sys

import pandas as pd

from trinet.config import CFG, ensure_dirs  # noqa: E402


def _find_raw_csv():
    """Locate the symptom CSV in the Kaggle download without hardcoding its exact filename.

    The dataset's file has been renamed across versions (e.g. 'DATA.csv',
    'MonkeyPox PATIENTS Dataset.csv'), so glob for it and fail loudly if it isn't unique.
    """
    csvs = sorted(CFG.symptom_dir.glob("*.csv"))
    if not csvs:
        sys.exit(
            f"[!] No symptom CSV under {CFG.symptom_dir}. Run `trinet download --only symptom`."
        )
    if len(csvs) > 1:
        print(f"[i] multiple CSVs in {CFG.symptom_dir}; using {csvs[0].name}")
    return csvs[0]


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(_find_raw_csv())
    print(f"[i] loaded {df.shape[0]} rows, columns: {list(df.columns)}")

    df = df.drop(columns=[c for c in CFG.symptom_drop if c in df.columns])
    df = df.dropna()

    # one-hot the only categorical column
    if "Systemic Illness" in df.columns:
        df = pd.concat(
            [df, pd.get_dummies(df["Systemic Illness"], prefix="SystemicIllness")], axis=1
        ).drop(columns=["Systemic Illness"])

    # map booleans / Positive-Negative to {0,1}
    df = df.replace(
        {True: 1, False: 0, "True": 1, "False": 0, "Positive": 1, "Negative": 0}
    ).infer_objects(copy=False)
    df = df.astype({c: int for c in df.columns if df[c].dtype != object})

    # NOTE: deliberately NO 'sum' feature (that was the leakage in the original).
    assert CFG.symptom_target in df.columns, "target column missing"
    pos = df[CFG.symptom_target].mean()
    out = CFG.data_proc / "symptom.csv"
    df.to_csv(out, index=False)
    print(f"[✓] cleaned symptom data -> {out}")
    print(
        f"    {df.shape[1] - 1} features, {df.shape[0]} rows, "
        f"class balance: {100 * pos:.1f}% positive"
    )
    print("    next: python -m src.models.symptom_model")


if __name__ == "__main__":
    main()

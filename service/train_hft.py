"""Train Hierarchical FastText and evaluate per-level accuracy."""
from __future__ import annotations

import pandas as pd

from src.categorization.hft import HierarchicalFastText
from src.data.hierarchy import load_offers_with_levels
from src.data.prepare_data import TextPreprocessor

NROWS = 50_000
MAX_LEVEL = 4
EMBED_DIM = 32
EPOCHS = 3
EVAL_N = 2000


def main() -> None:
    df = load_offers_with_levels(
        "data/raw_data/full_dataset.csv",
        "data/raw_data/category_mapping.csv",
        max_level=MAX_LEVEL,
    )
    df = (
        df.dropna(subset=["L1"])
        .sample(n=min(NROWS, len(df)), random_state=42)
        .reset_index(drop=True)
    )

    preprocessor = TextPreprocessor()
    df["t"] = df["text"].apply(preprocessor.normalize)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    level_cols = [f"L{i}" for i in range(1, MAX_LEVEL + 1) if f"L{i}" in train_df.columns]
    hft = HierarchicalFastText(
        max_level=MAX_LEVEL, embed_dim=EMBED_DIM, epochs=EPOCHS, min_count=2,
    )
    hft.fit(train_df["t"].tolist(), train_df[level_cols])

    # evaluate on test set
    test_eval = test_df.head(EVAL_N).reset_index(drop=True)
    preds = [hft.predict(t) for t in test_eval["t"].tolist()]

    violations = sum(1 for p in preds if p["violation_at"] is not None)
    total = len(preds)
    print(
        f"\nHierarchical consistency: {total - violations}/{total} consistent, "
        f"{violations} violated"
    )

    for level in range(1, MAX_LEVEL + 1):
        col = f"L{level}"
        if col not in test_eval.columns:
            break

        correct, total = 0, 0
        for i, pred in enumerate(preds):
            true_label = test_eval[col].iloc[i]
            if pd.isna(true_label):
                continue
            total += 1
            if len(pred["path"]) >= level and pred["path"][level - 1] == str(true_label):
                correct += 1

        if total > 0:
            print(f"  L{level} accuracy: {correct}/{total} = {correct / total:.2%}")


if __name__ == "__main__":
    main()

"""Flat FastText training + LIME demo on L1 categories."""
from __future__ import annotations

import pandas as pd

from src.data.prepare_data import TextPreprocessor
from src.evaluation.lime import LimeExplainer
from src.fasttext.model import FastText


def main() -> None:
    offers = pd.read_csv("data/raw_data/full_dataset.csv", sep="\t")
    categories = pd.read_csv("data/raw_data/category_mapping.csv", sep="\t")

    # take second segment as L1 (first is "Allegro")
    categories["L1"] = categories["category_name"].str.split(" > ").str[1]

    # merge on clean labels (noisy ones have wrong categories sometimes)
    offers = offers.merge(
        categories[["category_label", "L1"]],
        left_on="clean_category_id",
        right_on="category_label",
        how="left",
    ).dropna(subset=["L1"])

    print(f"Loaded {len(offers)} products across {offers['L1'].nunique()} categories\n")
    print("Category distribution (top 10):")
    for cat, count in offers["L1"].value_counts().head(10).items():
        print(f"  {cat:30s} {count:>6d}")

    preprocessor = TextPreprocessor()
    offers["processed"] = offers["text"].apply(preprocessor.normalize)

    # shuffle and 80/20 split
    shuffled = offers.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled[:split_idx]
    test_df = shuffled[split_idx:]

    train_texts = train_df["processed"].tolist()
    train_labels = train_df["L1"].tolist()
    test_texts = test_df["processed"].tolist()
    test_labels = test_df["L1"].tolist()

    print(f"\nSplit: {len(train_texts)} train / {len(test_texts)} test\n")
    print("Training FastText...")
    model = FastText(preprocessor, embed_dim=50, lr=0.05, epochs=5, min_count=2)
    model.fit(train_texts, train_labels)

    predictions = model.predict(test_texts)
    correct = sum(p == t for p, t in zip(predictions, test_labels))
    print(f"\nTest accuracy: {correct / len(test_labels):.2%} ({correct}/{len(test_labels)})")

    # show LIME explanations for a few examples
    explainer = LimeExplainer(model.predict_proba, num_samples=300, seed=42)

    print("\nLIME explanations for sample predictions\n")

    for i in range(3):
        text = test_texts[i]
        true_label = test_labels[i]
        pred_label = predictions[i]
        status = "correct" if true_label == pred_label else "WRONG"

        explanation = explainer.explain(text)
        top = LimeExplainer.top_features(explanation, k=5)

        print(f'"{text}"')
        print(f"  true: {true_label}, predicted: {pred_label}  [{status}]")
        print("  word importance:")
        for word, score in top:
            sign = "+" if score > 0 else "-"
            print(f"    {sign} {word:20s} ({abs(score):.3f})")
        print()


if __name__ == "__main__":
    main()

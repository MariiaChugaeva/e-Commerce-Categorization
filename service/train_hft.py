import pandas as pd

from src.categorization.hft import HierarchicalFastText
from src.data.hierarchy import load_offers_with_levels
from src.data.prepare_data import TextPreprocessor


NROWS = 50000
MAX_LEVEL = 4
EMBED_DIM = 32
EPOCHS = 3
EVAL_N = 2000

df = load_offers_with_levels(
    "data/raw_data/full_dataset.csv",
    "data/raw_data/category_mapping.csv",
    max_level=MAX_LEVEL,
)
df = df.dropna(subset=["L1"]).sample(n=min(NROWS, len(df)), random_state=42).reset_index(drop=True)

pre = TextPreprocessor()
df["t"] = df["text"].apply(pre.normalize)

split = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)

hft = HierarchicalFastText(max_level=MAX_LEVEL, embed_dim=EMBED_DIM, epochs=EPOCHS, min_count=2)
hft.fit(train_df["t"].tolist(), train_df[[f"L{i}" for i in range(1, MAX_LEVEL + 1) if f"L{i}" in train_df.columns]])

test_eval = test_df.head(EVAL_N).reset_index(drop=True)
preds = [hft.predict(t) for t in test_eval["t"].tolist()]

violations = sum(1 for p in preds if p["violation_at"] is not None)
print(f"\nhierarchical consistency: {len(preds) - violations}/{len(preds)} consistent, {violations} violated")

for lvl in range(1, MAX_LEVEL + 1):
    col = f"L{lvl}"
    if col not in test_eval.columns:
        break
    ok, tot = 0, 0
    for i, pred in enumerate(preds):
        true = test_eval[col].iloc[i]
        if pd.isna(true):
            continue
        tot += 1
        if len(pred["path"]) >= lvl and pred["path"][lvl - 1] == str(true):
            ok += 1
    if tot:
        print(f"L{lvl} accuracy: {ok}/{tot} = {ok / tot:.2%}")

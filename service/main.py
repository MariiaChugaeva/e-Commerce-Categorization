import pandas as pd
from src.data.prepare_data import TextPreprocessor
from src.fasttext.model import FastText
from src.evaluation.lime import LimeExplainer


# load both datasets, offers is the main one with product texts
offers = pd.read_csv("data/raw_data/full_dataset.csv", sep="\t")
categories = pd.read_csv("data/raw_data/category_mapping.csv", sep="\t")

# we only need L1, split by " > " to get it
categories["L1"] = categories["category_name"].str.split(" > ").str[1]

# merge to get category names into offers table
# using clean_category_id because noisy has wrong labels sometimes
offers = offers.merge(
    categories[["category_label", "L1"]],
    left_on="clean_category_id",
    right_on="category_label",
    how="left",
)

# drop rows where category wasnt found
offers = offers.dropna(subset=["L1"])
n_classes = offers["L1"].nunique()
print(f"loaded products across")
print()
print("сategory distribution (top 10):")
for cat, count in offers["L1"].value_counts().head(10).items():
    print(f"  {cat:30s} {count:>6d}")

# lowercase, remove punctuation etc
preprocessor = TextPreprocessor()
offers["processed"] = offers["text"].apply(preprocessor.normalize)

# shuffle and split 80/20, random_state for reproducibility
shuffled = offers.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(shuffled) * 0.8)
train_df = shuffled[:split]
test_df = shuffled[split:]

train_texts = train_df["processed"].tolist()
train_labels = train_df["L1"].tolist()
test_texts = test_df["processed"].tolist()
test_labels = test_df["L1"].tolist()

print()
print(f"split: {len(train_texts)} train / {len(test_texts)} test\n")
print("training FastText")

model = FastText(
    preprocessor,
    embed_dim=50,
    lr=0.05,
    epochs=5,
    min_count=2,
)
model.fit(train_texts, train_labels)

# simple accuracy check on test set
predictions = model.predict(test_texts)
correct = sum(p == l for p, l in zip(predictions, test_labels))
accuracy = correct / len(test_labels)
print(f"test accuracy: {accuracy:.2%} ({correct} / {len(test_labels)})")

# LIME to understand what words model pays attention
explainer = LimeExplainer(model.predict_proba, num_samples=300, seed=42)

print("\n\n\n")
print("LIME explanations for sample predictions")
print("\n\n\n")

# show 3 examples to see if model makes sense
for i in range(3):
    text = test_texts[i]
    true_label = test_labels[i]
    pred_label = predictions[i]
    match = "correct" if true_label == pred_label else "WRONG"

    exp = explainer.explain(text)
    top = LimeExplainer.top_features(exp, k=5)

    print(f'\n"{text}"')
    print(f"true: {true_label}, predicted: {pred_label}  [{match}]")
    print("word importance:")
    for word, score in top:
        direction = "+" if score > 0 else "-"
        print(f"{direction} {word:20s} ({abs(score):.3f})")

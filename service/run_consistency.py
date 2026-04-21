import json
import math
import os
import time
from statistics import mean

import numpy as np

from src.augmentation.ollama import OllamaClient, paraphrase_title
from src.categorization.hft import HierarchicalFastText
from src.data.hierarchy import load_offers_with_levels
from src.data.prepare_data import TextPreprocessor
from src.evaluation.consistency import path_agreement, score_correlation, top_k_overlap
from src.evaluation.lime import LimeExplainer


NROWS = 30000
N_SAMPLES = 8
N_VARIANTS = 3
LIME_SAMPLES = 200
LIME_LEVELS = (1, 2)
TOP_K = 5
OUT_JSON = "assets/consistency_results.json"

print("[data] loading")
df = load_offers_with_levels(
    "data/raw_data/full_dataset.csv",
    "data/raw_data/category_mapping.csv",
    max_level=4,
)
df = df.dropna(subset=["L1"]).sample(n=min(NROWS, len(df)), random_state=42).reset_index(drop=True)

pre = TextPreprocessor()
df["t"] = df["text"].apply(pre.normalize)
split = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)

print("[hft] training")
hft = HierarchicalFastText(max_level=4, embed_dim=32, epochs=3, min_count=2).fit(
    train_df["t"].tolist(),
    train_df[[c for c in ["L1", "L2", "L3", "L4"] if c in train_df.columns]],
)

test_sample = test_df.sample(n=min(N_SAMPLES, len(test_df)), random_state=7).reset_index(drop=True)

print(f"[augment] generating {N_VARIANTS} variants for {len(test_sample)} titles via Ollama")
client = OllamaClient()
augmented: list[dict] = []
t0 = time.time()
for i, row in test_sample.iterrows():
    orig = row["text"]
    try:
        variants = paraphrase_title(client, orig, n=N_VARIANTS)
    except Exception as e:
        print(f"  [warn] ollama failed on '{orig[:50]}': {e}")
        variants = []
    variants_norm = [pre.normalize(v) for v in variants if v.strip()]
    augmented.append({
        "original_raw": orig,
        "original": row["t"],
        "true_L1": row.get("L1"),
        "true_L2": row.get("L2"),
        "variants_raw": variants,
        "variants": variants_norm,
    })
    print(f"  [{i+1}/{len(test_sample)}] {len(variants_norm)} variants  ({time.time()-t0:.0f}s)")

print("[predict] running HFT on originals + variants")
for item in augmented:
    item["pred_original"] = hft.predict(item["original"])
    item["pred_variants"] = [hft.predict(v) for v in item["variants"]]

print("[lime] computing explanations")
per_level_stats: dict[int, dict] = {lvl: {"overlaps": [], "corrs": [], "n_pairs": 0} for lvl in LIME_LEVELS}
all_cases: list[dict] = []

for item in augmented:
    case = {
        "original": item["original_raw"],
        "original_normalized": item["original"],
        "variants": item["variants_raw"],
        "variants_normalized": item["variants"],
        "pred_original": item["pred_original"],
        "pred_variants": item["pred_variants"],
        "path_agreement_per_variant": [
            path_agreement(item["pred_original"]["path"], pv["path"])
            for pv in item["pred_variants"]
        ],
        "explanations": {},
    }
    for lvl in LIME_LEVELS:
        if lvl not in hft.models:
            continue
        proba_fn = lambda xs, L=lvl: hft.predict_proba_at(xs, L)
        lime = LimeExplainer(proba_fn, num_samples=LIME_SAMPLES, seed=0)
        orig_exp = lime.explain(item["original"])
        orig_label = orig_exp["label"]
        variant_exps = [lime.explain(v, label=orig_label) for v in item["variants"] if v]
        overlaps = [top_k_overlap(orig_exp, ve, k=TOP_K) for ve in variant_exps]
        corrs = [score_correlation(orig_exp, ve) for ve in variant_exps]
        per_level_stats[lvl]["overlaps"].extend(overlaps)
        per_level_stats[lvl]["corrs"].extend([c for c in corrs if not math.isnan(c)])
        per_level_stats[lvl]["n_pairs"] += len(overlaps)
        case["explanations"][f"L{lvl}"] = {
            "label": hft.models[lvl]._idx2label[orig_label],
            "original": {"words": orig_exp["words"], "scores": orig_exp["scores"].tolist()},
            "variants": [
                {"words": ve["words"], "scores": ve["scores"].tolist(), "overlap": o, "corr": c}
                for ve, o, c in zip(variant_exps, overlaps, corrs)
            ],
        }
    all_cases.append(case)

stability_rate_L1 = mean(
    sum(1 for pv in c["pred_variants"] if pv["path"] and c["pred_original"]["path"] and pv["path"][0] == c["pred_original"]["path"][0])
    / max(1, len(c["pred_variants"]))
    for c in all_cases
)

summary = {
    "n_samples": len(all_cases),
    "n_variants_total": sum(len(c["variants"]) for c in all_cases),
    "L1_prediction_stability": stability_rate_L1,
    "hft_violation_rate_originals": mean(1.0 if c["pred_original"]["violation_at"] is not None else 0.0 for c in all_cases),
    "hft_violation_rate_variants": mean(
        mean(1.0 if pv["violation_at"] is not None else 0.0 for pv in c["pred_variants"]) if c["pred_variants"] else 0.0
        for c in all_cases
    ),
    "per_level": {
        f"L{lvl}": {
            "n_pairs": per_level_stats[lvl]["n_pairs"],
            "mean_top_k_overlap": mean(per_level_stats[lvl]["overlaps"]) if per_level_stats[lvl]["overlaps"] else None,
            "mean_score_correlation": mean(per_level_stats[lvl]["corrs"]) if per_level_stats[lvl]["corrs"] else None,
        }
        for lvl in LIME_LEVELS
    },
}

os.makedirs("assets", exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump({"summary": summary, "cases": all_cases}, f, indent=2, default=str)

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print(f"\nwrote {OUT_JSON}")

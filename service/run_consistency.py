"""Consistency experiment: paraphrase titles via Ollama, compare HFT predictions + LIME stability."""
from __future__ import annotations

import json
import math
import os
import time
from statistics import mean
from typing import Any

import numpy as np

from src.augmentation.ollama import OllamaClient, paraphrase_title
from src.categorization.hft import HierarchicalFastText
from src.data.hierarchy import load_offers_with_levels
from src.data.prepare_data import TextPreprocessor
from src.evaluation.consistency import path_agreement, score_correlation, top_k_overlap
from src.evaluation.lime import LimeExplainer

NROWS = 30_000
N_SAMPLES = 8
N_VARIANTS = 3
LIME_SAMPLES = 200
LIME_LEVELS = (1, 2)
TOP_K = 5
OUT_JSON = "assets/consistency_results.json"


def _train_model(preprocessor: TextPreprocessor) -> tuple[HierarchicalFastText, Any, Any]:
    print("[data] loading")
    df = load_offers_with_levels(
        "data/raw_data/full_dataset.csv",
        "data/raw_data/category_mapping.csv",
        max_level=4,
    )
    df = (
        df.dropna(subset=["L1"])
        .sample(n=min(NROWS, len(df)), random_state=42)
        .reset_index(drop=True)
    )
    df["t"] = df["text"].apply(preprocessor.normalize)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    level_cols = [c for c in ["L1", "L2", "L3", "L4"] if c in train_df.columns]

    print("[hft] training")
    hft = HierarchicalFastText(max_level=4, embed_dim=32, epochs=3, min_count=2)
    hft.fit(train_df["t"].tolist(), train_df[level_cols])
    return hft, preprocessor, test_df


def _augment(test_df: Any, preprocessor: TextPreprocessor) -> list[dict]:
    test_sample = (
        test_df.sample(n=min(N_SAMPLES, len(test_df)), random_state=7)
        .reset_index(drop=True)
    )

    print(f"[augment] generating {N_VARIANTS} variants for {len(test_sample)} titles via Ollama")
    client = OllamaClient()
    augmented: list[dict] = []
    t0 = time.time()

    for i, row in test_sample.iterrows():
        original = row["text"]
        try:
            variants = paraphrase_title(client, original, n=N_VARIANTS)
        except Exception as exc:
            print(f"  [warn] ollama failed on '{original[:50]}': {exc}")
            variants = []

        variants_norm = [preprocessor.normalize(v) for v in variants if v.strip()]
        augmented.append({
            "original_raw": original,
            "original": row["t"],
            "true_L1": row.get("L1"),
            "true_L2": row.get("L2"),
            "variants_raw": variants,
            "variants": variants_norm,
        })
        elapsed = time.time() - t0
        print(f"  [{i + 1}/{len(test_sample)}] {len(variants_norm)} variants ({elapsed:.0f}s)")

    return augmented


def _evaluate(hft: HierarchicalFastText, augmented: list[dict]) -> tuple[dict, list[dict]]:
    print("[predict] running HFT on originals + variants")
    for item in augmented:
        item["pred_original"] = hft.predict(item["original"])
        item["pred_variants"] = [hft.predict(v) for v in item["variants"]]

    print("[lime] computing explanations")
    per_level: dict[int, dict] = {
        lvl: {"overlaps": [], "corrs": [], "n_pairs": 0} for lvl in LIME_LEVELS
    }
    all_cases: list[dict] = []

    for item in augmented:
        case: dict[str, Any] = {
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

            def proba_fn(xs: list[str], level: int = lvl) -> np.ndarray:
                return hft.predict_proba_at(xs, level)

            lime = LimeExplainer(proba_fn, num_samples=LIME_SAMPLES, seed=0)
            orig_exp = lime.explain(item["original"])
            orig_label = orig_exp["label"]
            variant_exps = [
                lime.explain(v, label=orig_label) for v in item["variants"] if v
            ]

            overlaps = [top_k_overlap(orig_exp, ve, k=TOP_K) for ve in variant_exps]
            corrs = [score_correlation(orig_exp, ve) for ve in variant_exps]

            per_level[lvl]["overlaps"].extend(overlaps)
            per_level[lvl]["corrs"].extend(c for c in corrs if not math.isnan(c))
            per_level[lvl]["n_pairs"] += len(overlaps)

            case["explanations"][f"L{lvl}"] = {
                "label": hft.models[lvl]._idx2label[orig_label],
                "original": {
                    "words": orig_exp["words"],
                    "scores": orig_exp["scores"].tolist(),
                },
                "variants": [
                    {
                        "words": ve["words"],
                        "scores": ve["scores"].tolist(),
                        "overlap": o,
                        "corr": c,
                    }
                    for ve, o, c in zip(variant_exps, overlaps, corrs)
                ],
            }

        all_cases.append(case)

    # how often does L1 prediction stay the same across paraphrases
    stability_l1 = mean(
        sum(
            1 for pv in c["pred_variants"]
            if pv["path"] and c["pred_original"]["path"]
            and pv["path"][0] == c["pred_original"]["path"][0]
        ) / max(1, len(c["pred_variants"]))
        for c in all_cases
    )

    summary = {
        "n_samples": len(all_cases),
        "n_variants_total": sum(len(c["variants"]) for c in all_cases),
        "L1_prediction_stability": stability_l1,
        "hft_violation_rate_originals": mean(
            1.0 if c["pred_original"]["violation_at"] is not None else 0.0
            for c in all_cases
        ),
        "hft_violation_rate_variants": mean(
            mean(
                1.0 if pv["violation_at"] is not None else 0.0
                for pv in c["pred_variants"]
            ) if c["pred_variants"] else 0.0
            for c in all_cases
        ),
        "per_level": {
            f"L{lvl}": {
                "n_pairs": per_level[lvl]["n_pairs"],
                "mean_top_k_overlap": (
                    mean(per_level[lvl]["overlaps"])
                    if per_level[lvl]["overlaps"] else None
                ),
                "mean_score_correlation": (
                    mean(per_level[lvl]["corrs"])
                    if per_level[lvl]["corrs"] else None
                ),
            }
            for lvl in LIME_LEVELS
        },
    }

    return summary, all_cases


def main() -> None:
    preprocessor = TextPreprocessor()
    hft, preprocessor, test_df = _train_model(preprocessor)
    augmented = _augment(test_df, preprocessor)
    summary, all_cases = _evaluate(hft, augmented)

    os.makedirs("assets", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"summary": summary, "cases": all_cases}, f, indent=2, default=str)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()

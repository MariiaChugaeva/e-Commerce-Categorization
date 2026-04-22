from __future__ import annotations

import math
from typing import List

import numpy as np


def _top_k_words(explanation: dict, k: int) -> set[str]:
    words = explanation["words"]
    scores = np.asarray(explanation["scores"], dtype=float)
    if len(words) == 0:
        return set()
    order = np.argsort(np.abs(scores))[::-1][:k]
    return {words[i] for i in order}


def top_k_overlap(exp_a: dict, exp_b: dict, k: int = 5) -> float:
    """Jaccard overlap of top-k important words between two explanations."""
    a = _top_k_words(exp_a, k)
    b = _top_k_words(exp_b, k)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def score_correlation(exp_a: dict, exp_b: dict) -> float:
    """Pearson correlation of LIME scores for words that appear in both explanations."""
    scores_a = dict(zip(exp_a["words"], exp_a["scores"]))
    scores_b = dict(zip(exp_b["words"], exp_b["scores"]))
    common = sorted(set(scores_a) & set(scores_b))

    if len(common) < 2:
        return math.nan

    arr_a = np.array([scores_a[w] for w in common], dtype=float)
    arr_b = np.array([scores_b[w] for w in common], dtype=float)

    if arr_a.std() == 0 or arr_b.std() == 0:
        return math.nan
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def path_agreement(path_a: List[str], path_b: List[str]) -> int:
    """How many levels match from the root before first disagreement."""
    count = 0
    for a, b in zip(path_a, path_b):
        if a == b:
            count += 1
        else:
            break
    return count

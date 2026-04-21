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
    a = _top_k_words(exp_a, k)
    b = _top_k_words(exp_b, k)
    if not a and not b:
        return 1.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)


def score_correlation(exp_a: dict, exp_b: dict) -> float:
    sa = {w: s for w, s in zip(exp_a["words"], exp_a["scores"])}
    sb = {w: s for w, s in zip(exp_b["words"], exp_b["scores"])}
    common = list(set(sa) & set(sb))
    if len(common) < 2:
        return math.nan
    a = np.array([sa[w] for w in common], dtype=float)
    b = np.array([sb[w] for w in common], dtype=float)
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def path_agreement(path_a: List[str], path_b: List[str]) -> int:
    n = 0
    for x, y in zip(path_a, path_b):
        if x == y:
            n += 1
        else:
            break
    return n

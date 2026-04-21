import numpy as np
from typing import Callable, List, Sequence


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(scores))[::-1][:k]


def _join(words: Sequence[str], keep_mask: np.ndarray) -> str:
    return " ".join(w for w, m in zip(words, keep_mask) if m)


def comprehensiveness(
    predict_proba_fn: Callable[[List[str]], np.ndarray],
    words: List[str],
    scores: np.ndarray,
    label: int,
    k: int = 5,
) -> float:
    top = _top_k_indices(scores, k)
    keep = np.ones(len(words), dtype=bool)
    keep[top] = False
    probs = predict_proba_fn([" ".join(words), _join(words, keep)])
    return float(probs[0, label] - probs[1, label])


def sufficiency(
    predict_proba_fn: Callable[[List[str]], np.ndarray],
    words: List[str],
    scores: np.ndarray,
    label: int,
    k: int = 5,
) -> float:
    top = _top_k_indices(scores, k)
    keep = np.zeros(len(words), dtype=bool)
    keep[top] = True
    probs = predict_proba_fn([" ".join(words), _join(words, keep)])
    return float(probs[0, label] - probs[1, label])


def monotonicity(
    predict_proba_fn: Callable[[List[str]], np.ndarray],
    words: List[str],
    scores: np.ndarray,
    label: int,
) -> float:
    n = len(words)
    if n < 2:
        return 0.0
    order = np.argsort(np.abs(scores))[::-1]
    texts = [" ".join(words)]
    for idx in order:
        keep = np.ones(n, dtype=bool)
        keep[idx] = False
        texts.append(_join(words, keep))
    probs = predict_proba_fn(texts)
    drops = probs[0, label] - probs[1:, label]
    importance = np.abs(scores[order])
    if importance.std() == 0 or drops.std() == 0:
        return 0.0
    return float(np.corrcoef(importance, drops)[0, 1])


def accuracy_at_k(probs: np.ndarray, true_label_ids: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = np.any(topk == true_label_ids[:, None], axis=1)
    return float(hits.mean())

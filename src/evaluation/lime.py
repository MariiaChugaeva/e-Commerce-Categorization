from __future__ import annotations

import numpy as np
from typing import Callable, List


class LimeExplainer:
    """Simplified LIME for text (Ribeiro et al., 2016).

    Randomly masks words to create perturbed inputs, gets model predictions
    on them, then fits a weighted linear model to get per-word importance.
    """

    def __init__(
        self,
        predict_proba_fn: Callable[[List[str]], np.ndarray],
        num_samples: int = 500,
        seed: int = 42,
    ) -> None:
        self.predict_proba_fn = predict_proba_fn
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def _perturb(self, words: List[str]) -> tuple[np.ndarray, List[str]]:
        n = len(words)
        masks = np.ones((self.num_samples, n), dtype=int)

        for i in range(1, self.num_samples):
            mask = self.rng.integers(0, 2, size=n)
            # make sure at least one word survives
            if mask.sum() == 0:
                mask[self.rng.integers(0, n)] = 1
            masks[i] = mask

        texts = [
            " ".join(w for w, m in zip(words, row) if m)
            for row in masks
        ]
        return masks, texts

    @staticmethod
    def _cosine_distance(masks: np.ndarray) -> np.ndarray:
        """Distance between the original (first row) and each perturbation."""
        original = masks[0].astype(float)
        norm_orig = np.linalg.norm(original)
        distances = np.zeros(len(masks))

        for i in range(len(masks)):
            row = masks[i].astype(float)
            norm_row = np.linalg.norm(row)
            if norm_orig == 0 or norm_row == 0:
                distances[i] = 1.0
            else:
                distances[i] = 1.0 - np.dot(row, original) / (norm_orig * norm_row)

        return distances

    @staticmethod
    def _kernel(distances: np.ndarray, width: float = 0.25) -> np.ndarray:
        """Exponential kernel — closer perturbations get higher weight."""
        return np.exp(-(distances ** 2) / (width ** 2))

    @staticmethod
    def _weighted_least_squares(
        X: np.ndarray, y: np.ndarray, weights: np.ndarray,
    ) -> np.ndarray:
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, np.newaxis]
        yw = y * sqrt_w

        # small regularization to avoid singular matrix
        XtX = Xw.T @ Xw + np.eye(Xw.shape[1]) * 1e-6
        Xty = Xw.T @ yw
        return np.linalg.solve(XtX, Xty)

    def explain(self, text: str, label: int | None = None) -> dict:
        words = text.split()
        if len(words) == 0:
            return {"words": [], "scores": np.array([]), "label": label, "label_proba": 0.0}

        masks, perturbed_texts = self._perturb(words)
        proba = self.predict_proba_fn(perturbed_texts)

        # if no label given, explain whatever the model predicts
        if label is None:
            label = int(proba[0].argmax())

        target = proba[:, label]
        distances = self._cosine_distance(masks)
        weights = self._kernel(distances)
        coefficients = self._weighted_least_squares(masks, target, weights)

        return {
            "words": words,
            "scores": coefficients,
            "label": label,
            "label_proba": float(proba[0][label]),
        }

    @staticmethod
    def top_features(explanation: dict, k: int = 10) -> List[tuple[str, float]]:
        """Top-k words sorted by absolute importance."""
        words = explanation["words"]
        scores = explanation["scores"]
        order = np.argsort(np.abs(scores))[::-1][:k]
        return [(words[i], float(scores[i])) for i in order]

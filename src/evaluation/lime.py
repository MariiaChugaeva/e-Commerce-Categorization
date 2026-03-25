import numpy as np
from typing import Callable, List


class LimeExplainer:
    """simplified LIME implementation for text classification
    based on the original paper by Ribeiro et al. but without all the extra stuff
    we only need basic word-level explanations here"""

    def __init__(
        self,
        predict_proba_fn: Callable[[List[str]], np.ndarray],
        num_samples: int = 500,
        seed: int = 42,
    ):
        self.predict_proba_fn = predict_proba_fn
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def _perturb(self, words: List[str]) -> tuple[np.ndarray, List[str]]:
        """randomly mask words to create perturbed versions of the text"""
        n = len(words)
        masks = np.ones((self.num_samples, n), dtype=int)
        masks[0] = 1

        for i in range(1, self.num_samples):
            mask = self.rng.integers(0, 2, size=n)
            # make sure at least one word survives, otherwise predict_proba breaks
            if mask.sum() == 0:
                mask[self.rng.integers(0, n)] = 1
            masks[i] = mask

        texts = []
        for mask in masks:
            texts.append(" ".join(w for w, m in zip(words, mask) if m))
        return masks, texts

    @staticmethod
    def _cosine_distance(masks: np.ndarray) -> np.ndarray:
        """cosine distance between original mask and each perturbed mask
        closer perturbations should have more weight in the regression"""
        original = masks[0].astype(float)
        norm_orig = np.linalg.norm(original)

        distances = np.zeros(len(masks))
        for i in range(len(masks)):
            row = masks[i].astype(float)
            norm_row = np.linalg.norm(row)

            if norm_orig == 0 or norm_row == 0:
                distances[i] = 1.0
                continue

            dot = np.dot(row, original)
            cosine_sim = dot / (norm_orig * norm_row)
            distances[i] = 1.0 - cosine_sim

        return distances

    @staticmethod
    def _kernel(distances: np.ndarray, width: float = 0.25) -> np.ndarray:
        """exponential kernel to convert distances into weights"""
        squared = distances ** 2
        scaled = squared / (width ** 2)
        return np.exp(-scaled)

    @staticmethod
    def _weighted_least_squares(
        X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """weighted linear regression to get feature importances"""
        sqrt_w = np.sqrt(weights)
        X_w = X * sqrt_w[:, np.newaxis]
        y_w = y * sqrt_w

        XtX = X_w.T @ X_w
        XtX += np.eye(XtX.shape[0]) * 1e-6  # regularization to avoid singular matrix
        Xty = X_w.T @ y_w

        coefs = np.linalg.solve(XtX, Xty)
        return coefs

    def explain(self, text: str, label: int = None) -> dict:
        """takes text and returns word-level importance scores
        if label is None we explain the predicted class"""
        words = text.split()
        if len(words) == 0:
            return {"words": [], "scores": np.array([])}

        masks, perturbed_texts = self._perturb(words)
        proba = self.predict_proba_fn(perturbed_texts)

        # if no label specified, explain whatever the model thinks is most likely
        if label is None:
            label = proba[0].argmax()

        target = proba[:, label]
        distances = self._cosine_distance(masks)
        weights = self._kernel(distances)
        coefs = self._weighted_least_squares(masks, target, weights)

        return {
            "words": words,
            "scores": coefs,
            "label": label,
            "label_proba": proba[0][label],
        }

    @staticmethod
    def top_features(explanation: dict, k: int = 10) -> List[tuple[str, float]]:
        """get top-k most important words sorted by absolute score"""
        words = explanation["words"]
        scores = explanation["scores"]
        order = np.argsort(np.abs(scores))[::-1][:k]
        return [(words[i], scores[i]) for i in order]

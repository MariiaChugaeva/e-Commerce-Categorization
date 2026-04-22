from __future__ import annotations

import numpy as np
from typing import List

from src.data.prepare_data import TextPreprocessor


def _softmax(logits: np.ndarray) -> np.ndarray:
    # clip to avoid overflow, then shift for numerical stability
    clipped = np.clip(logits, -30, 30)
    shifted = clipped - clipped.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class FastText:
    """Our own FastText-style classifier from scratch.

    Averages word embeddings, passes through a linear layer, then softmax.
    Trained with SGD on cross-entropy loss.
    """

    def __init__(
        self,
        preprocessor: TextPreprocessor,
        embed_dim: int = 100,
        lr: float = 0.1,
        epochs: int = 5,
        min_count: int = 1,
        seed: int = 42,
    ) -> None:
        self.preprocessor = preprocessor
        self.embed_dim = embed_dim
        self.lr = lr
        self.epochs = epochs
        self.min_count = min_count
        self.seed = seed

        self._label2idx: dict[str, int] = {}
        self._idx2label: list[str] = []
        self._embedding: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None

    def _build_labels(self, labels: List[str]) -> np.ndarray:
        unique = sorted(set(labels))
        self._label2idx = {label: idx for idx, label in enumerate(unique)}
        self._idx2label = unique
        return np.array([self._label2idx[label] for label in labels])

    def _forward(self, word_ids: List[int]) -> tuple[np.ndarray, np.ndarray]:
        # if no known words, just use zeros
        if len(word_ids) == 0:
            hidden = np.zeros(self.embed_dim)
        else:
            hidden = self._embedding[word_ids].mean(axis=0)
        logits = hidden @ self._weights + self._bias
        return hidden, _softmax(logits)

    def _train_step(self, word_ids: List[int], label_id: int) -> float:
        hidden, probs = self._forward(word_ids)

        # gradient of cross-entropy w.r.t. softmax output
        grad_output = probs.copy()
        grad_output[label_id] -= 1.0

        self._weights -= self.lr * np.outer(hidden, grad_output)
        self._bias -= self.lr * grad_output

        # backprop into embeddings, scale by number of words
        if len(word_ids) > 0:
            grad_embed = (grad_output @ self._weights.T) / len(word_ids)
            self._embedding[word_ids] -= self.lr * grad_embed

        return -np.log(probs[label_id] + 1e-10)

    def fit(self, texts: List[str], labels: List[str]) -> FastText:
        self.preprocessor.build_vocab(texts, min_count=self.min_count)
        label_ids = self._build_labels(labels)

        rng = np.random.default_rng(self.seed)
        n_classes = len(self._idx2label)
        vocab_size = self.preprocessor.vocab_size

        # small random init for embeddings, zeros for weights
        self._embedding = rng.normal(0, 1.0 / self.embed_dim, (vocab_size, self.embed_dim))
        self._weights = np.zeros((self.embed_dim, n_classes))
        self._bias = np.zeros(n_classes)

        # encode once so we don't re-tokenize every epoch
        encoded = [self.preprocessor.encode(t) for t in texts]
        n_samples = len(texts)

        for epoch in range(self.epochs):
            order = rng.permutation(n_samples)
            total_loss = sum(self._train_step(encoded[i], label_ids[i]) for i in order)
            avg_loss = total_loss / n_samples
            print(f"  epoch {epoch + 1}/{self.epochs}  loss={avg_loss:.4f}", flush=True)

        return self

    def predict(self, texts: List[str]) -> List[str]:
        results = []
        for text in texts:
            word_ids = self.preprocessor.encode(text)
            _, probs = self._forward(word_ids)
            results.append(self._idx2label[probs.argmax()])
        return results

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns probability distribution over classes — needed for LIME."""
        all_probs = []
        for text in texts:
            word_ids = self.preprocessor.encode(text)
            _, probs = self._forward(word_ids)
            all_probs.append(probs)
        return np.array(all_probs)

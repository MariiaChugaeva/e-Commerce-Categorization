import numpy as np
from typing import List

from src.data.prepare_data import TextPreprocessor


def _softmax(logits: np.ndarray) -> np.ndarray:
    """numerically stable softmax, clip to avoid overflow with big logits"""
    clipped = np.clip(logits, -30, 30)
    e = np.exp(clipped - clipped.max())
    return e / e.sum()


class FastText:
    """our own fasttext implementation from scratch
    basically averages word embeddings and passes through linear layer
    not as fast as facebook's version but easier to debug and modify"""

    def __init__(
        self,
        preprocessor: TextPreprocessor,
        embed_dim: int = 100,
        lr: float = 0.1,
        epochs: int = 5,
        min_count: int = 1,
        seed: int = 42,
    ):
        self.preprocessor = preprocessor
        self.embed_dim = embed_dim
        self.lr = lr
        self.epochs = epochs
        self.min_count = min_count
        self.seed = seed

        self._label2idx: dict = {}
        self._idx2label: list = []
        self._embedding: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None

    def _build_labels(self, labels: List) -> np.ndarray:
        """create mapping between label strings and integer indices"""
        unique = sorted(set(labels))
        self._label2idx = {l: i for i, l in enumerate(unique)}
        self._idx2label = unique
        return np.array([self._label2idx[l] for l in labels])

    def _forward(self, word_ids: List[int]) -> tuple[np.ndarray, np.ndarray]:
        """forward pass: average embeddings -> linear -> softmax
        if text is empty (no known words) we just use zero vector"""
        if len(word_ids) == 0:
            hidden = np.zeros(self.embed_dim)
        else:
            hidden = self._embedding[word_ids].mean(axis=0)
        logits = hidden @ self._weights + self._bias
        return hidden, _softmax(logits)

    def _train_one(self, word_ids: List[int], label: int) -> float:
        """single training step with SGD
        returns cross-entropy loss for this sample"""
        hidden, probs = self._forward(word_ids)

        # gradient of cross-entropy with softmax is just
        grad = probs.copy()
        grad[label] -= 1.0

        self._weights -= self.lr * np.outer(hidden, grad)
        self._bias -= self.lr * grad

        # update embeddings too, divide by num words to keep gradient scale reasonable
        if len(word_ids) > 0:
            embed_grad = (grad @ self._weights.T) / len(word_ids)
            self._embedding[word_ids] -= self.lr * embed_grad

        return -np.log(probs[label] + 1e-10)

    def fit(self, texts: List[str], labels: List) -> "FastText":
        """train the model on given texts and labels
        builds vocab first, then runs SGD for specified number of epochs"""
        self.preprocessor.build_vocab(texts, min_count=self.min_count)
        label_ids = self._build_labels(labels)

        rng = np.random.default_rng(self.seed)
        n_classes = len(self._idx2label)
        vocab_size = self.preprocessor.vocab_size

        # init embeddings with small random values, weights and bias start at zero
        self._embedding = rng.normal(0, 1.0 / self.embed_dim, (vocab_size, self.embed_dim))
        self._weights = np.zeros((self.embed_dim, n_classes))
        self._bias = np.zeros(n_classes)

        # pre-encode all texts so we dont redo tokenization every epoch
        encoded = [self.preprocessor.encode(t) for t in texts]
        n = len(texts)

        for epoch in range(self.epochs):
            order = rng.permutation(n)  # shuffle each epoch
            total_loss = 0.0
            for i in order:
                total_loss += self._train_one(encoded[i], label_ids[i])
            print(f"epoch {epoch + 1}/{self.epochs}  loss={total_loss / n:.4f}")

        return self

    def predict(self, texts: List[str]) -> List:
        """predict class labels for list of texts"""
        results = []
        for text in texts:
            word_ids = self.preprocessor.encode(text)
            _, probs = self._forward(word_ids)
            results.append(self._idx2label[probs.argmax()])
        return results

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """same as predict but returns probability distribution over all classes
        needed for LIME explainer"""
        all_probs = []
        for text in texts:
            word_ids = self.preprocessor.encode(text)
            _, probs = self._forward(word_ids)
            all_probs.append(probs)
        return np.array(all_probs)

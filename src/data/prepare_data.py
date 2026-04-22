from __future__ import annotations

import re
from typing import List

import nltk
import pandas as pd
from nltk.stem import SnowballStemmer, WordNetLemmatizer


class TextPreprocessor:
    """Handles text cleaning and tokenization.

    Optionally applies stemming (stem_type="l1") or
    lemmatization (stem_type="l2") via NLTK.
    """

    def __init__(self, stem_type: str | None = None) -> None:
        self.stem_type = stem_type
        self.word2idx: dict[str, int] = {}
        self.vocab_size: int = 0
        self.stemmer: SnowballStemmer | None = None
        self.lemmatizer: WordNetLemmatizer | None = None

        if stem_type in ("l1", "l2"):
            nltk.download("punkt", quiet=True)

        if stem_type == "l1":
            self.stemmer = SnowballStemmer("english")
        elif stem_type == "l2":
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            self.lemmatizer = WordNetLemmatizer()

    def normalize(self, text: str) -> str:
        """Lowercase, remove punctuation, collapse whitespace."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _apply_morphology(self, word: str) -> str:
        if self.stemmer is not None:
            return self.stemmer.stem(word)
        if self.lemmatizer is not None:
            return self.lemmatizer.lemmatize(word)
        return word

    def tokenize(self, text: str) -> List[str]:
        normalized = self.normalize(text)
        if not normalized:
            return []
        return [self._apply_morphology(token) for token in normalized.split()]

    def to_fasttext_string(self, text: str) -> str:
        return self.normalize(text)

    def build_vocab(self, texts: List[str], min_count: int = 1) -> None:
        """Count words, keep only those with freq >= min_count."""
        counts: dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        kept = [word for word, count in counts.items() if count >= min_count]
        self.word2idx = {word: idx for idx, word in enumerate(kept)}
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str) -> List[int]:
        """Convert text to list of word indices, skip unknown words."""
        return [self.word2idx[t] for t in self.tokenize(text) if t in self.word2idx]

    def preprocess_df(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        df = df.copy()
        df["normalized_text"] = df[text_col].apply(self.normalize)
        df["tokens"] = df[text_col].apply(self.tokenize)
        df["fasttext_text"] = df[text_col].apply(self.to_fasttext_string)
        return df

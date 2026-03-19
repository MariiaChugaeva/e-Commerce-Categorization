import re
import pandas as pd
from typing import List
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer


class TextPreprocessor:
    def __init__(self, stem_type: str | None = None):
        self.stem_type = stem_type
        self.word2idx: dict[str, int] = {}
        self.vocab_size: int = 0

        if stem_type in ("l1", "l2"):
                nltk.download("punkt", quiet=True)
                if stem_type == "l2":
                    nltk.download("wordnet", quiet=True)
                    nltk.download("omw-1.4", quiet=True)

        if stem_type == "l1":
                self.stemmer = SnowballStemmer("english")
                self.lemmatizer = None
        elif stem_type == "l2":
            self.stemmer = None
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stemmer = None
            self.lemmatizer = None

    def normalize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r"\s+", " ", text).strip()

    def _stem_or_lemmatize(self, word: str) -> str:
        if self.stem_type == "l1" and self.stemmer:
            return self.stemmer.stem(word)
        if self.stem_type == "l2" and self.lemmatizer:
            return self.lemmatizer.lemmatize(word)
        return word

    def tokenize(self, text: str) -> List[str]:
        normalized = self.normalize(text)
        if not normalized:
            return []
        tokens = normalized.split()
        return [self._stem_or_lemmatize(t) for t in tokens]

    def to_fasttext_string(self, text: str) -> str:
        return self.normalize(text)

    def build_vocab(self, texts: List[str], min_count: int = 1) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        words = [w for w, c in counts.items() if c >= min_count]
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str) -> List[int]:
        return [self.word2idx[t] for t in self.tokenize(text) if t in self.word2idx]

    def preprocess_df(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        df = df.copy()
        df["normalized_text"] = df[text_col].apply(self.normalize)
        df["tokens"] = df[text_col].apply(self.tokenize)
        df["fasttext_text"] = df[text_col].apply(self.to_fasttext_string)
        return df
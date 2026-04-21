from typing import List

import numpy as np
import pandas as pd

from src.data.prepare_data import TextPreprocessor
from src.fasttext.model import FastText


class HierarchicalFastText:
    def __init__(
        self,
        max_level: int = 4,
        embed_dim: int = 32,
        lr: float = 0.05,
        epochs: int = 3,
        min_count: int = 2,
        seed: int = 42,
        stem_type: str | None = None,
    ):
        self.max_level = max_level
        self.ft_params = {
            "embed_dim": embed_dim,
            "lr": lr,
            "epochs": epochs,
            "min_count": min_count,
            "seed": seed,
        }
        self.stem_type = stem_type
        self.models: dict[int, FastText] = {}
        self.valid_children: dict[int, dict[str, set[str]]] = {}

    def fit(self, texts: List[str], labels_df: pd.DataFrame) -> "HierarchicalFastText":
        texts_arr = np.asarray(texts, dtype=object)
        for i in range(1, self.max_level + 1):
            col = f"L{i}"
            if col not in labels_df.columns:
                break
            series = labels_df[col].reset_index(drop=True)
            mask = series.notna() & (series.astype(str).str.strip() != "")
            if mask.sum() < 10:
                break
            subset_texts = texts_arr[mask.values].tolist()
            subset_labels = series[mask].tolist()
            print(f"[HFT] level {i}: {len(subset_texts)} samples, {len(set(subset_labels))} classes")
            pre = TextPreprocessor(stem_type=self.stem_type)
            model = FastText(pre, **self.ft_params).fit(subset_texts, subset_labels)
            self.models[i] = model
        self._build_hierarchy(labels_df)
        return self

    def _build_hierarchy(self, df: pd.DataFrame) -> None:
        for i in range(2, self.max_level + 1):
            col_p, col_c = f"L{i-1}", f"L{i}"
            if col_c not in df.columns or col_p not in df.columns:
                break
            pairs = df[[col_p, col_c]].dropna()
            children: dict[str, set[str]] = {}
            for parent, child in zip(pairs[col_p].astype(str), pairs[col_c].astype(str)):
                children.setdefault(parent, set()).add(child)
            self.valid_children[i] = children

    def predict(self, text: str) -> dict:
        path: list[str] = []
        violation_at: int | None = None
        for i in range(1, self.max_level + 1):
            if i not in self.models:
                break
            pred = self.models[i].predict([text])[0]
            if i > 1:
                allowed = self.valid_children.get(i, {}).get(path[-1], set())
                if pred not in allowed:
                    violation_at = i
                    break
            path.append(pred)
        return {"path": path, "violation_at": violation_at}

    def predict_proba_at(self, texts: List[str], level: int) -> np.ndarray:
        return self.models[level].predict_proba(texts)

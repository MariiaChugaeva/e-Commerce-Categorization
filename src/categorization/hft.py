from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data.prepare_data import TextPreprocessor
from src.fasttext.model import FastText


class HierarchicalFastText:
    """One FastText model per hierarchy level.

    At inference, each level's prediction is constrained to be a valid
    child of the parent predicted at the previous level. If the constraint
    is violated, prediction stops early.
    """

    def __init__(
        self,
        max_level: int = 4,
        embed_dim: int = 32,
        lr: float = 0.05,
        epochs: int = 3,
        min_count: int = 2,
        seed: int = 42,
        stem_type: str | None = None,
    ) -> None:
        self.max_level = max_level
        self.ft_params = dict(
            embed_dim=embed_dim, lr=lr, epochs=epochs,
            min_count=min_count, seed=seed,
        )
        self.stem_type = stem_type
        self.models: dict[int, FastText] = {}
        self.valid_children: dict[int, dict[str, set[str]]] = {}

    def fit(
        self, texts: List[str], labels_df: pd.DataFrame,
    ) -> HierarchicalFastText:
        texts_arr = np.asarray(texts, dtype=object)

        for level in range(1, self.max_level + 1):
            col = f"L{level}"
            if col not in labels_df.columns:
                break

            series = labels_df[col].reset_index(drop=True)
            mask = series.notna() & (series.astype(str).str.strip() != "")
            if mask.sum() < 10:
                break

            subset_texts = texts_arr[mask.values].tolist()
            subset_labels = series[mask].tolist()
            n_classes = len(set(subset_labels))
            print(f"[HFT] level {level}: {len(subset_texts)} samples, {n_classes} classes", flush=True)

            preprocessor = TextPreprocessor(stem_type=self.stem_type)
            model = FastText(preprocessor, **self.ft_params).fit(subset_texts, subset_labels)
            self.models[level] = model

        self._build_hierarchy(labels_df)
        return self

    def _build_hierarchy(self, df: pd.DataFrame) -> None:
        """Build parent->children mapping from the training data."""
        for level in range(2, self.max_level + 1):
            parent_col, child_col = f"L{level - 1}", f"L{level}"
            if child_col not in df.columns or parent_col not in df.columns:
                break

            pairs = df[[parent_col, child_col]].dropna()
            children: dict[str, set[str]] = {}
            for parent, child in zip(
                pairs[parent_col].astype(str), pairs[child_col].astype(str),
            ):
                children.setdefault(parent, set()).add(child)
            self.valid_children[level] = children

    def predict(self, text: str) -> dict:
        path: list[str] = []
        violation_at: int | None = None

        for level in range(1, self.max_level + 1):
            if level not in self.models:
                break

            pred = self.models[level].predict([text])[0]

            # check hierarchy constraint: predicted child must be valid
            if level > 1:
                allowed = self.valid_children.get(level, {}).get(path[-1], set())
                if pred not in allowed:
                    violation_at = level
                    break

            path.append(pred)

        return {"path": path, "violation_at": violation_at}

    def predict_proba_at(self, texts: List[str], level: int) -> np.ndarray:
        return self.models[level].predict_proba(texts)

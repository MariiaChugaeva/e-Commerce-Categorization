"""Web service for interactive LIME exploration.

Trains HierarchicalFastText on startup, then exposes an API
to classify product titles and show LIME explanations.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.categorization.hft import HierarchicalFastText
from src.data.hierarchy import load_offers_with_levels
from src.data.prepare_data import TextPreprocessor
from src.evaluation.lime import LimeExplainer

NROWS = 50_000
MAX_LEVEL = 4
EMBED_DIM = 32
EPOCHS = 3
LIME_SAMPLES = 300

# global state filled on startup
_state: dict[str, Any] = {}


def _train() -> None:
    print("[web] Loading data...", flush=True)
    df = load_offers_with_levels(
        "data/raw_data/full_dataset.csv",
        "data/raw_data/category_mapping.csv",
        max_level=MAX_LEVEL,
    )
    df = (
        df.dropna(subset=["L1"])
        .sample(n=min(NROWS, len(df)), random_state=42)
        .reset_index(drop=True)
    )

    preprocessor = TextPreprocessor()
    df["t"] = df["text"].apply(preprocessor.normalize)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    level_cols = [f"L{i}" for i in range(1, MAX_LEVEL + 1) if f"L{i}" in train_df.columns]

    print("[web] Training HierarchicalFastText...", flush=True)
    hft = HierarchicalFastText(
        max_level=MAX_LEVEL, embed_dim=EMBED_DIM, epochs=EPOCHS, min_count=2,
    )
    hft.fit(train_df["t"].tolist(), train_df[level_cols])

    _state["hft"] = hft
    _state["preprocessor"] = preprocessor
    _state["test_df"] = test_df
    _state["sample_titles"] = (
        test_df["text"].dropna().sample(n=min(50, len(test_df)), random_state=7).tolist()
    )
    print("[web] Model ready.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _train()
    yield


app = FastAPI(title="LIME Explorer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    lime_level: int = 1
    num_samples: int = LIME_SAMPLES
    top_k: int = 10


class WordScore(BaseModel):
    word: str
    score: float


class LevelPrediction(BaseModel):
    level: int
    label: str


class PredictResponse(BaseModel):
    normalized_text: str
    path: list[LevelPrediction]
    violation_at: int | None
    lime_label: str
    lime_label_proba: float
    lime_level: int
    word_scores: list[WordScore]


@app.get("/api/samples")
def get_samples() -> dict:
    return {"titles": _state.get("sample_titles", [])}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    hft: HierarchicalFastText = _state["hft"]
    preprocessor: TextPreprocessor = _state["preprocessor"]

    normalized = preprocessor.normalize(req.text)
    result = hft.predict(normalized)

    path = [
        LevelPrediction(level=i + 1, label=label)
        for i, label in enumerate(result["path"])
    ]

    # clamp lime_level to what we actually have
    lime_level = min(req.lime_level, max(hft.models.keys()))
    if lime_level not in hft.models:
        lime_level = 1

    def proba_fn(texts: list[str]) -> np.ndarray:
        return hft.predict_proba_at(texts, lime_level)

    explainer = LimeExplainer(proba_fn, num_samples=req.num_samples, seed=42)
    explanation = explainer.explain(normalized)
    top = LimeExplainer.top_features(explanation, k=req.top_k)

    label_name = hft.models[lime_level]._idx2label[explanation["label"]]

    return PredictResponse(
        normalized_text=normalized,
        path=path,
        violation_at=result["violation_at"],
        lime_label=label_name,
        lime_label_proba=explanation["label_proba"],
        lime_level=lime_level,
        word_scores=[WordScore(word=w, score=round(s, 4)) for w, s in top],
    )


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")

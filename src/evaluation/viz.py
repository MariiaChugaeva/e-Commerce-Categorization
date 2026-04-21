import numpy as np
import matplotlib.pyplot as plt

_POS = "#2a7ab0"
_NEG = "#c0392b"


def plot_word_importance(explanation: dict, k: int = 10, ax=None, title: str | None = None):
    words = explanation["words"]
    scores = np.asarray(explanation["scores"], dtype=float)
    if len(words) == 0:
        return ax
    order = np.argsort(np.abs(scores))[::-1][:k]
    top_words = [words[i] for i in order][::-1]
    top_scores = scores[order][::-1]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(2.0, 0.4 * len(top_words))))

    colors = [_POS if s > 0 else _NEG for s in top_scores]
    ax.barh(range(len(top_words)), top_scores, color=colors)
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("LIME score")
    if title is not None:
        ax.set_title(title, fontsize=10)
    return ax

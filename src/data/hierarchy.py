from __future__ import annotations

import pandas as pd


def load_offers_with_levels(
    offers_path: str,
    mapping_path: str,
    max_level: int = 4,
    label_column: str = "clean_category_id",
) -> pd.DataFrame:
    """Load offers and join with hierarchy levels from the category mapping.

    Category paths look like "Allegro > Electronics > Phones > ...".
    Segment 0 is "Allegro" (marketplace root), so L1 starts from segment 1.
    """
    offers = pd.read_csv(offers_path, sep="\t")
    mapping = pd.read_csv(mapping_path, sep="\t")

    parts = mapping["category_name"].str.split(" > ", expand=True)
    levels = pd.DataFrame({"category_label": mapping["category_label"]})
    for i in range(1, max_level + 1):
        if i in parts.columns:
            levels[f"L{i}"] = parts[i]

    merged = offers.merge(
        levels,
        left_on=label_column,
        right_on="category_label",
        how="left",
    )
    return merged

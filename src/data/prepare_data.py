import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "data" / "raw_data"
    proc_dir = project_root / "data" / "prepared_data"

    full_df = pd.read_csv(raw_dir / "full_dataset.csv", sep="\t")
    cat_map = pd.read_csv(raw_dir / "category_mapping.csv", sep="\t")


    levels = cat_map["category_name"].str.split(" > ", expand=True)
    levels.columns = [f"L{i+1}" for i in range(levels.shape[1])]
    cat_map = pd.concat([cat_map, levels], axis=1)

    cat_indexed = cat_map.set_index("category_label")

    full_df = full_df.merge(
        cat_indexed.add_suffix("_clean"),
        left_on="clean_category_id",
        right_index=True,
        how="left",
    )

    full_df = full_df.merge(
        cat_indexed.add_suffix("_noisy"),
        left_on="noisy_category_id",
        right_index=True,
        how="left",
    )

    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / "processed_dataset.parquet"
    full_df.to_parquet(out_path, index=False)
    print(f"Saved processed dataset to {out_path}")


if __name__ == "__main__":
    main()
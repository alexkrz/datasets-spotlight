import json
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
from PIL import Image
from renumics import spotlight

CAST_COLUMN = False


def main(
    data_p: Path = Path.home() / "Data" / "FairnessBias" / "bupt_balance-parquet" / "bupt_balance.parquet",  # fmt: skip
):
    assert data_p.exists()
    # ds = datasets.load_dataset("parquet", data_files=str(data_p), split="train")
    ds = datasets.Dataset.from_parquet(str(data_p), split="train")
    print("Number of items:", len(ds))
    print(ds.features)

    if CAST_COLUMN and data_p.stem == "bupt_balance":
        # Add ethnicity names as ClassLabel
        with open(data_p.parent / "ethnicity_labels.json", "r") as f:
            ethnicity_labels = json.load(f)
        ethnicity_names = list(ethnicity_labels.values())
        ds = ds.cast_column("ethnicity", datasets.ClassLabel(names=ethnicity_names))
        print(ds.features)

    entry = ds[0]
    print("Keys:", entry.keys())
    img = entry["img"]
    label = entry["label"]
    print("Image properties:", img)
    # print("Label:", label)

    # plt.imshow(img)
    # plt.xlabel(label)
    # plt.show()

    spotlight.show(ds)


if __name__ == "__main__":
    main()

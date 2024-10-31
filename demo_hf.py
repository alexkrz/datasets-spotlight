import json
from pathlib import Path

import datasets
import pandas as pd
from renumics import spotlight


def main(out_dir: Path = Path.home() / "Data" / "EmotionRecognition" / "affectnethq"):
    # df = pd.read_csv("https://renumics.com/data/mnist/mnist-tiny.csv")
    # print(df.head())
    # spotlight.show(df, dtype={"image": spotlight.Image})

    ds = datasets.load_dataset("Piro17/affectnethq", split="train")

    # CORRECTION: It looks like the labels are already included in the .parquet file.
    # OLD: Unfortunately, currently the label mapping is not included in the parquet file.
    # Instead, we should export the labels as a separate .json file
    print(ds.features)
    class_label: datasets.ClassLabel = ds.features["label"]
    class_names = class_label.names
    class_dict = {idx: name for idx, name in enumerate(class_names)}

    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_dict, f, indent=4)
        f.write("\n")  # Add a trailing newline

    # Here we save the dataset as a single .parquet file
    ds.to_parquet(out_dir / "affectnethq.parquet")

    # spotlight.show(ds)


if __name__ == "__main__":
    main()

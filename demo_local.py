from pathlib import Path

import datasets
import matplotlib.pyplot as plt
from PIL import Image
from renumics import spotlight


def main(
    data_p: Path = Path.home() / "Data" / "TrainDatasets" / "parquet-files" / "ms1mv2.parquet",
):
    assert data_p.exists()
    # ds = datasets.load_dataset("parquet", data_files=str(data_p), split="train")
    ds = datasets.Dataset.from_parquet(str(data_p), split="train")
    print("Number of items:", len(ds))

    entry = ds[0]
    print("Keys:", entry.keys())
    img = entry["img"]
    label = entry["label"]
    print("Image properties:", img)

    # plt.imshow(img)
    # plt.xlabel(label)
    # plt.show()

    # spotlight.show(ds)


if __name__ == "__main__":
    main()

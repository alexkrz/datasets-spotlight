from pathlib import Path

import datasets
import matplotlib.pyplot as plt
from PIL import Image
from renumics import spotlight


def main(data_p: Path = Path.home() / "Data" / "TrainDatasets" / "casia_webface-parquet"):
    assert data_p.exists()
    ds = datasets.load_dataset("parquet", data_dir=data_p, split="train")
    print("Number of items:", len(ds))

    # img, label = ds[0].values()
    # plt.imshow(img)
    # plt.xlabel(label)
    # plt.show()

    spotlight.show(ds)


if __name__ == "__main__":
    main()

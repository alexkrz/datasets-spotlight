import datasets
import pandas as pd
from renumics import spotlight

# df = pd.read_csv("https://renumics.com/data/mnist/mnist-tiny.csv")
# print(df.head())
# spotlight.show(df, dtype={"image": spotlight.Image})

ds = datasets.load_dataset("cifar10", split="test")
spotlight.show(ds)

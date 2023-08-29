import pandas as pd
import numpy as np

df = pd.read_csv("../data/Deduplication_Example_10.csv")
df.columns = ["TEXT"]

d = ["I went to the supermarket and I bought an apple and an orange."] * 1900
df2 = pd.DataFrame(data=d, columns=["TEXT"])

df3 = pd.concat([df, df2], axis=0, ignore_index=True)

df3.to_csv(path_or_buf="../data/Deduplication_Example_10-2000.csv")

a=1

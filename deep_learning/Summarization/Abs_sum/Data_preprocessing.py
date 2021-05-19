import pandas as pd
import random
import os


#join train sentence pairs together into dataframe
data = pd.concat([pd.read_csv(f'{RAW_DATA}sumdata/train/train.article.txt', sep="\n"),
                  pd.read_csv(f'{RAW_DATA}sumdata/train/train.title.txt', sep="\n")], axis=1)
data.columns = ["article", "title"]
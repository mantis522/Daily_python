import pandas as pd
import random
import os


#join train sentence pairs together into dataframe
data = pd.concat([pd.read_csv(r"D:\ruin\data\summarization\summary\sumdata\train\train.article.txt\train.article.txt", sep="\n"),
                  pd.read_csv(r"D:\ruin\data\summarization\summary\sumdata\train\train.title.txt\train.title.txt", sep="\n")], axis=1)
data.columns = ["article", "title"]

val = pd.concat([pd.read_csv(r"D:\ruin\data\summarization\summary\sumdata\train\valid.article.filter.txt", sep="\n"),
                  pd.read_csv(r"D:\ruin\data\summarization\summary\sumdata\train\valid.title.filter.txt", sep="\n")], axis=1)
val.columns = ["article", "title"]

print(len(data), len(val))

for i in range(5):
    r = random.randint(0, 50)
    print(data.iloc[r]['article'])
    print(data.iloc[r]['title'])

#save train, val datasets
data.to_csv(r"D:\ruin\data\summarization\summary\csv\train_ds.csv", index=None)
val.to_csv(r"D:\ruin\data\summarization\summary\csv\valid_ds.csv", index=None)

sample_train = data.sample(80000)
sample_val = val.sample(20000)

#save sample train, val, and test datasets
sample_train.to_csv(r'D:\ruin\data\summarization\summary\csv\sample_train_ds.csv', index=None)
sample_val.to_csv(r'D:\ruin\data\summarization\summary\csv\sample_valid_ds.csv', index=None)


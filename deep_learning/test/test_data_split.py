import pandas as pd
import re
from sklearn.model_selection import train_test_split

imdb_dataset = pd.read_csv(r"D:\ruin\data\IMDB Dataset.csv")

def preprocessing(dataset):
    txt_list = []
    label_list = []
    for a in range(len(dataset)):
        txt = imdb_dataset['review'][a]
        label = imdb_dataset['sentiment'][a]
        if label == 'positive':
            label = 1
        elif label == 'negative':
            label = 0
        txt = re.sub(r'\<br />', '', txt)
        txt_list.append(txt)
        label_list.append(label)

    df = pd.DataFrame([x for x in zip(txt_list, label_list)])
    df.columns = ['text', 'label']
    return df

imdb_dataset = preprocessing(imdb_dataset)

X_train, X_test = train_test_split(imdb_dataset, test_size=0.2, random_state=123)
X_train.to_csv(r"D:\ruin\data\csv_file\imdb_split\train_data.csv", index=False)
X_test.to_csv(r"D:\ruin\data\csv_file\imdb_split\test_data.csv", index=False)
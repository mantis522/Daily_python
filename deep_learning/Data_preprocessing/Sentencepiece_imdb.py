import pandas as pd
import sentencepiece as spm
import csv
import json

vocab_file = r"D:\ruin\data\transformer_imdb\imdb.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

# infile = r"D:\ruin\data\IMDB Dataset2.csv"
# df = pd.read_csv(infile)
# print(df['Unnamed: 0'])

def prepare_train(vocab, infile, outfile):
  df = pd.read_csv(infile)
  with open(outfile, 'w') as f:
    for index, row in df.iterrows():
      document = row['text']
      if type(document) != str:
        continue
      instance = {"doc": vocab.encode_as_pieces(document), "label": row["label"]}
      f.write(json.dumps(instance))
      f.write("\n")

prepare_train(vocab, r"D:\ruin\data\csv_file\imdb_split\train_data.csv", "D:/ruin/data/ratings_train.json")
prepare_train(vocab, r"D:\ruin\data\csv_file\imdb_split\test_data.csv", "D:/ruin/data/ratings_test.json")


from transformers import pipeline
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
# summarizer = pipeline("summarization")
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

imdb_data = r"D:\ruin\data\IMDB Dataset2.csv"

df_imdb = pd.read_csv(imdb_data)

test = df_imdb['text'][49999]

print(df_imdb)

summary_text = summarizer(test, max_length=100, min_length=5, do_sample=False)[0]['summary_text']

print("original text : ", test)

print("summarized_text : ", summary_text)
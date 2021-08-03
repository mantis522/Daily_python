from transformers import pipeline
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# using pipeline API for summarization task
summarization = pipeline("summarization")
imdb_data = r"D:\ruin\data\IMDB Dataset2.csv"

df_imdb = pd.read_csv(imdb_data)

original_text = df_imdb['text'][10000]


summary_text = summarization(original_text)[0]['summary_text']
print("original:", original_text)
print("Summary:", summary_text)


## https://www.thepythoncode.com/article/text-summarization-using-huggingface-transformers-python
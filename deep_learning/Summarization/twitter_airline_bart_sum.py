from transformers import pipeline
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization")

airline = r"D:\ruin\data\Twitter_US_Airline\original\airline_tweet_train.csv"

df_airline = pd.read_csv(airline)

original_text = df_airline['text'][4]
summarized_text = summarizer(original_text, max_length=380, min_length=5, do_sample=False)[0]['summary_text']

print('original_text : ',original_text)
print('summarized_text : ',summarized_text)
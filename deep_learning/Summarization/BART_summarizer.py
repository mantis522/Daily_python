from transformers import pipeline
import os

## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization")

## To use the t5-base model for summarization:
## summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

text = """But while they may agree that the virus is here to stay, in some form, for the foreseeable future, these countries have radically different approaches to dealing with it.
Singapore, an island state of 5.69 million, and the UK, home to an estimated 66 million people, have had very different pandemic experiences -- and outcomes -- so far.
While the UK has one of the highest numbers of Covid-19 related deaths in the world -- nearly 129,000 since the pandemic started -- only 36 people have died of Covid-19 in Singapore. For every 100,000 of the population in the UK, there have been 192.64 Covid-19 deaths in the UK. This goes down to 0.63 in Singapore, according to Johns Hopkins University data."""

summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)


### https://towardsdatascience.com/abstractive-summarization-using-pytorch-f5063e67510
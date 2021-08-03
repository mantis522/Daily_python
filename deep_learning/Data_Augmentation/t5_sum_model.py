from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

imdb_data = r"D:\ruin\data\IMDB Dataset2.csv"
df_imdb = pd.read_csv(imdb_data)

original_text = df_imdb['text'][10000]

# encode the text into tensor of integers using the appropriate tokenizer
inputs = tokenizer.encode(original_text, return_tensors="pt", max_length=512, truncation=True)

# generate the summarization output
outputs = model.generate(
    inputs,
    max_length=150,
    min_length=40,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True)
# just for debugging

print("original_text : ", original_text)
print("output_text : ", tokenizer.decode(outputs[0]))
import sentencepiece as spm

# corpus = r"D:\ruin\data\imdb_review.txt"
# prefix = "imdb"
# vocab_size = 8000
# spm.SentencePieceTrainer.train(
#     f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
#     " --model_type=bpe" +
#     " --max_sentence_length=999999" + # 문장 최대 길이
#     " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
#     " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
#     " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
#     " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
#     " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰

import pandas as pd
import json

vocab_file = r"D:\ruin\data\transformer_test\imdb.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

def prepare_train(vocab, infile, outfile):
    df = pd.read_csv(infile)
    with open(outfile, "w") as f:
        for index, row in df.iterrows():
            document = row["text"]
            if type(document) != str:
                continue
            instance = { "id": row["Unnamed: 0"], "doc": vocab.encode_as_pieces(document), "label": row["label"] }
            f.write(json.dumps(instance))
            f.write("\n")

prepare_train(vocab, r"D:\ruin\data\transformer_test\imdb_train.csv", "ratings_train.json")
prepare_train(vocab, r"D:\ruin\data\transformer_test\imdb_test.csv", "ratings_test.json")
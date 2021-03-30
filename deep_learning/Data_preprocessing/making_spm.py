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

sp = spm.SentencePieceProcessor()
vocab_file = r"D:\ruin\data\transformer_imdb\imdb.model"
sp.load(vocab_file)

lines = [
  "I didn't at all think of it this way.",
  "I have waited a long time for someone to film"
]
for line in lines:
  print(line)
  print(sp.encode_as_pieces(line))
  print(sp.encode_as_ids(line))
  print()

print(sp.GetPieceSize())
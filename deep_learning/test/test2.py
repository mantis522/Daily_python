import sentencepiece as spm

vocab_file = r"D:\ruin\data\transformer_test\kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)


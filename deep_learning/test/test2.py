import argparse
from collections import Counter, OrderedDict

from prenlp.tokenizer import SentencePiece, NLTKMosesTokenizer

TOKENIZER = {'nltk_moses': NLTKMosesTokenizer()}
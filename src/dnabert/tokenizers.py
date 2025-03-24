from itertools import product
import re
from typing import Iterable

class DnaTokenizer:
    """
    This tokenizer interface is based on Huggingface's tokenizers.
    """
    def __init__(
        self,
        kmer: int = 1,
        kmer_stride: int = 1,
        normalize_sequences: bool = True
    ):
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.vocab = self.get_vocab()
        self.normalize_sequences = normalize_sequences
        self._normalize_re = re.compile(r"[^A-Z]")

    def convert_to_token_ids(self, tokens: Iterable[str]):
        return list(map(self.vocab.get, tokens))

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False):
        raise NotImplementedError()

    def normalize(self, sequence: str):
        return re.sub(self._normalize_re, "", sequence.upper())

    def tokenize(self, sequence: str):
        n = len(sequence)
        tokens = []
        for i in range(self.kmer, n+1, self.kmer_stride):
            token = sequence[i-self.kmer:i]
            if token not in self.vocab:
                tokens.append("[UNK]")
            else:
                tokens.append(token)
        return tokens

    def get_vocab(self):
        identifiers = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4
        }
        for kmer in product("ACGT", repeat=self.kmer):
            token = "".join(kmer)
            identifiers[token] = len(identifiers)
        return identifiers

    def __len__(self):
        return len(self.vocab)

    @property
    def num_token_ids(self):
        return max(self.vocab.values()) + 1

    def __call__(self, sequence: str):
        if self.normalize_sequences:
            sequence = self.normalize(sequence)
        tokens = self.tokenize(sequence)
        token_ids = self.convert_to_token_ids(tokens)
        return token_ids
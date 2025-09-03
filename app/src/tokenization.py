# tokenization.py
from typing import List, Dict

class SimpleTokenizer:
    """
    A very simple tokenizer and vocabulary builder.

    What this does:
    - Takes in plain English sentences.
    - Splits them into words using spaces ("the cat sat").
    - Assigns each unique word an ID number (word → number).
    - Adds special tokens:
        [PAD] = used to fill empty space when sentences are shorter.
        [UNK] = used for words not seen before (unknown words).
    - Provides methods to convert sentences into lists of numbers (encode),
      and numbers back into words (decode).

    Why we need this:
    Transformers can only work with numbers, not raw text.
    This is the "translator" between human words and machine numbers.
    """

    def __init__(self, sentences: List[str]):
        """
        Build the vocabulary from a list of sentences.
        """
        tokens = set()
        for sent in sentences:
            tokens.update(sent.lower().split())
        self.vocab = {tok: i+2 for i, tok in enumerate(sorted(tokens))}
        self.vocab["[PAD]"] = 0
        self.vocab["[UNK]"] = 1
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

    def encode(self, sentence: str, max_len: int = 20) -> List[int]:
        """
        Turn a human sentence into a list of numbers.
        Example: "the cat sat" → [5, 8, 2, 0, 0, 0...] (padded to max_len).
        """
        ids = [self.vocab.get(tok, self.vocab["[UNK]"]) for tok in sentence.lower().split()]
        ids = ids[:max_len] + [self.vocab["[PAD]"]] * max(0, max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Turn numbers back into words (the reverse of encode).
        """
        return " ".join(self.inv_vocab.get(i, "[UNK]") for i in ids if i != self.vocab["[PAD]"])

    def vocab_size(self) -> int:
        """
        Get the size of the vocabulary (how many words + special tokens).
        """
        return len(self.vocab)

# embeddings.py
import math
import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding

class TokenEmbedding(nn.Module):
    """
    Converts word IDs into embeddings and adds position information.

    Steps:
    1. Each word ID is looked up in an Embedding table (like a dictionary),
       producing a dense vector that represents the word's meaning.
    2. A positional encoding is added, so the model knows word order.
    3. The result is fed into the Transformer.

    Input: [5, 8, 2]  (IDs for "the cat sat")
    Output: [[0.1, -0.3, ...], [0.7, 0.2, ...], [0.5, -0.8, ...]]
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.scale = math.sqrt(d_model)  # scaling helps training stability

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Convert input_ids into embeddings with position added.
        """
        x = self.token_embed(input_ids) * self.scale
        return self.pos_encoding(x)

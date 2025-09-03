# positional_encoding.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Adds position information to word embeddings.

    Problem:
    - Transformers do not know the order of words in a sentence.
      For them, "cat sat on mat" and "mat sat on cat" look the same.
    
    Solution:
    - Positional Encoding adds patterns of sine and cosine waves
      to word embeddings.
    - These patterns are unique for each position (1st word, 2nd word, etc.),
      so the model learns "who is where".

    Example:
    Input: embedding for "cat" at position 2
    Output: embedding + position pattern (so model knows it's the 2nd word).
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # saved inside model, not trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)

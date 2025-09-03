# encoder.py
import torch.nn as nn
from embeddings import TokenEmbedding
from encoder_layer import EncoderLayer

class Encoder(nn.Module):
    """
    The full Transformer Encoder (stack of layers).

    Components:
    - Embedding: turns word IDs into vectors + adds position info.
    - N Encoder Layers: each does Attention + FeedForward processing.
    - LayerNorm: final cleanup for stability.

    Use case:
    - Converts a human sentence (after tokenization) into a sequence
      of embeddings where each word understands its context.

    Example:
    Input: "the cat sat on the mat"
    Output: 6 vectors (one for each word), each containing context.
    """

    def __init__(self, num_layers: int, vocab_size: int, d_model: int,
                 num_heads: int, dim_ff: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        """
        Pass input IDs through the encoder.
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len), 1=keep, 0=ignore
        """
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)

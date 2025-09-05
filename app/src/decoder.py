# decoder.py
import torch.nn as nn
from embeddings import TokenEmbedding
from decoder_layer import DecoderLayer

class Decoder(nn.Module):
    """
    The full Transformer Decoder (stack of layers).

    Components:
    - Embedding: turns target word IDs into vectors (with position info).
    - N Decoder Layers: each does masked self-attention,
      cross-attention with encoder output, and feed-forward.
    - LayerNorm: final cleanup.

    Use case:
    - Given encoder output (source sentence context) and target tokens so far,
      predicts the next token in sequence.
    """

    def __init__(self, num_layers: int, vocab_size: int, d_model: int,
                 num_heads: int, dim_ff: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt_ids, enc_out, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt_ids: (batch, tgt_len) target sentence IDs
        enc_out: (batch, src_len, d_model) encoder output
        tgt_mask: (tgt_len, tgt_len) mask to prevent looking ahead
        tgt_key_padding_mask: (batch, tgt_len) padding mask for targets
        memory_key_padding_mask: (batch, src_len) padding mask for source
        """
        x = self.embedding(tgt_ids)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.norm(x)

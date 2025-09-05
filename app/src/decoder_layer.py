# decoder_layer.py
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    A single block of the Transformer Decoder.

    Steps inside this layer:
    1. Masked Self-Attention:
       - The decoder can only look at *past words* (not future).
       - This ensures predictions are made left-to-right.
    2. Add & Normalize.
    3. Cross-Attention:
       - The decoder looks at the encoder output (context from input sentence).
       - This is how translation works: "dog" in source → "chien" in target.
    4. Add & Normalize.
    5. Feed-Forward Network (extra processing).
    6. Add & Normalize again.
    """

    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        # masked self-attention (decoder attends to itself, causal)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross-attention (decoder attends to encoder output)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # feed-forward network
        self.ff = PositionwiseFeedForward(d_model, dim_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        x: decoder input (batch, tgt_len, d_model)
        enc_out: encoder output (batch, src_len, d_model)
        tgt_mask: (tgt_len, tgt_len) causal mask (prevents peeking ahead)
        tgt_key_padding_mask: padding mask for decoder inputs
        memory_key_padding_mask: padding mask for encoder outputs
        """
        # 1. Masked self-attention
        self_attn_out, _ = self.self_attn(x, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 2. Cross-attention
        # reuse MultiHeadAttention but pass encoder output as "values"
        cross_out, _ = self.cross_attn(enc_out, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. Feed forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x

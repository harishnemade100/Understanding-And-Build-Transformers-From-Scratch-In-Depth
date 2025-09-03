# encoder_layer.py
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    A single block of the Transformer Encoder.

    Steps inside this layer:
    1. Multi-Head Attention:
       - Each word looks at other words for context.
    2. Add & Normalize:
       - Add original + attention output, then normalize for stability.
    3. Feed-Forward Network:
       - Process each word independently with a small neural net.
    4. Add & Normalize again.

    This structure is repeated N times to build a full Encoder.
    """

    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, dim_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        Forward pass through one encoder layer.
        """
        mha_out, _ = self.mha(x, key_padding_mask)
        x = self.norm1(x + self.dropout(mha_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

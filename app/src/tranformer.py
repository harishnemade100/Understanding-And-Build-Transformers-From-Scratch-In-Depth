# transformer.py
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    """
    Full Transformer model (Encoder + Decoder).

    Flow:
    1. Encoder processes the source sentence → context vectors.
    2. Decoder generates target sentence, one word at a time,
       using both its past words and encoder output.
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, num_heads: int = 8, dim_ff: int = 2048,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(num_encoder_layers, src_vocab_size, d_model,
                               num_heads, dim_ff, max_len, dropout)
        self.decoder = Decoder(num_decoder_layers, tgt_vocab_size, d_model,
                               num_heads, dim_ff, max_len, dropout)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_ids, tgt_ids,
                src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None):
        """
        src_ids: (batch, src_len) source sentence IDs
        tgt_ids: (batch, tgt_len) target sentence IDs
        """
        enc_out = self.encoder(src_ids, attention_mask=src_padding_mask)
        dec_out = self.decoder(tgt_ids, enc_out, tgt_mask, tgt_padding_mask, src_padding_mask)
        logits = self.out_proj(dec_out)  # (batch, tgt_len, vocab_size)
        return logits

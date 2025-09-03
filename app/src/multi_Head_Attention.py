import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: the core of Transformers.

    What it does:
    - Each word looks at other words in the sentence.
    - For example, in "the cat sat on the mat":
        "cat" should pay attention to "sat" (verb),
        "mat" should pay attention to "on" (preposition).
    - Multi-head means it does this in multiple "views" (heads),
      so the model can learn different types of relationships.

    Steps:
    1. Project words into Q (query), K (key), V (value) vectors.
    2. Compute attention scores: Q • K^T.
    3. Apply softmax → get attention weights.
    4. Multiply weights with V (values).
    5. Combine all heads together.

    Input: word embeddings (batch, seq_len, d_model)
    Output: new embeddings where each word has context about others.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        b, t, d = x.size()
        x = x.view(b, t, self.num_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        b, h, t, d = x.size()
        return x.permute(0, 2, 1, 3).reshape(b, t, h*d)

    def forward(self, x, key_padding_mask=None):
        """
        Forward pass of attention.
        - x: input embeddings (batch, seq_len, d_model)
        - key_padding_mask: (batch, seq_len), True where we should ignore tokens (like [PAD]).
        Returns:
        - output embeddings (batch, seq_len, d_model)
        - attention weights (batch, heads, seq_len, seq_len)
        """
        b, t, d = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(self._split_heads, (q, k, v))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        out = self._merge_heads(context)
        return self.out_proj(out), attn

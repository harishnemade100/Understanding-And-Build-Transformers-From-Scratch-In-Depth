# feed_forward.py
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    The "thinking" part of the Transformer layer.

    After attention mixes words together,
    this feed-forward network transforms each word’s representation
    into something richer.

    Steps:
    - Linear layer expands dimensions (more features).
    - Apply GELU activation (non-linearity).
    - Dropout (for regularization).
    - Linear layer reduces back to original size.

    Think of it as "extra processing" for each word individually.
    """

    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Transform embeddings word by word.
        """
        return self.net(x)

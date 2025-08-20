import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # stability
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q,K,V: [seq, d_k]  (single example, single head for simplicity)
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)     # [seq, seq]
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)
    A = softmax(scores, axis=-1)          # [seq, seq]
    return A @ V, A                       # [seq, d_k], [seq, seq]

# example
seq, d_model, d_k = 5, 16, 16
X = np.random.randn(seq, d_model)

# simple learned-like projections (random for demo)
Wq = np.random.randn(d_model, d_k)
Wk = np.random.randn(d_model, d_k)
Wv = np.random.randn(d_model, d_k)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

out, attn = scaled_dot_product_attention(Q, K, V)  # out: [5,16], attn: [5,5]
print(out.shape, attn.shape)

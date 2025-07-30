import numpy as np

@staticmethod
def softmax(scores):
    np_scores = np.array(scores)
    exp_scores = np.exp(np_scores - np.max(np_scores))

    sum_scores = np.sum(exp_scores)
    softmax_scores = exp_scores / sum_scores
    return softmax_scores.tolist()



sentence = ["I", "love", "AI"]
word_embeddings = {
    "I":    np.array([1, 0, 1, 0]),
    "love": np.array([0, 1, 0, 1]),
    "AI":   np.array([1, 1, 0, 0])
}

seq_dim = len(sentence)
d_model = len(next(iter(word_embeddings.values())))
num_heads = 2

head_dim = d_model // num_heads


X = np.array(word for word in sentence if word in word_embeddings)

W_q = np.random.rand(d_model, head_dim)
W_k = np.random.rand(d_model, head_dim)
W_v = np.random.rand(d_model, head_dim)


Q = np.dot(X, W_q)
K = np.dot(X, W_k)
V = np.dot(X, W_v)

# Step 1: Calculate attention scores
scores = np.dot(Q, K.T) / np.sqrt(head_dim)
weight = cls.softmax(scores)



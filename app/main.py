from src.embeddings import EmbeddingLayer

if __name__ == "__main__":
    sentences = ["I", "love", "pizza"]
    embeddings = EmbeddingLayer.get_word_embeddings(sentences)

    # print("Embeddings Shape:", embeddings.shape)
    print("Embeddings:\n", embeddings)  # Random values; they will change on each run
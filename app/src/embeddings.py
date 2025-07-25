import torch 
import torch.nn as nn
from src.positional_Encoding import PositionalEncoding

class EmbeddingLayer(nn.Module):
    # Step 1: Function to build vocab from any sentence
    @classmethod
    def build_vocab(cls, sentences):
        vocab = {word: idx for idx, word in enumerate(sorted(set(sentences)))}
        return vocab

    @classmethod
    # Step 2: Function to convert words to IDs using the vocab
    def sentence_to_ids(cls, sentence, vocab):
        return [vocab[word] for word in sentence]

    # Step 3: Main function to get embeddings from any sentence
    @classmethod
    def get_word_embeddings(cls, sentences, embedding_dim=8):
        # Step 1: Build vocabulary
        vocab = cls.build_vocab(sentences)
        vocab_size = len(vocab)

           # Step 2: Convert sentence to tensor of word IDs
        input_ids = torch.tensor([cls.sentence_to_ids(sentences, vocab)])  # Shape: (1, seq_len)

        # Step 3: Create the embedding layer
        embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Step 4: Pass input_ids to embedding layer
        embeddings = embedding_layer(input_ids)  # Shape: (1, seq_len, embedding_dim)

        embeddings_dict = {word: embeddings[0][i].tolist() for i, word in enumerate(sentences)}


        print("Vocabulary:", vocab)
        print("Input IDs:", input_ids.tolist())
        print("Embeddings Shape:", embeddings.shape)
        print("Embeddings:\n", embeddings_dict)

        return PositionalEncoding.position_encoding(sentences, embeddings_dict, embedding_dim)
import math
import numpy as np



class PositionalEncoding:
    @staticmethod
    def get_position_encoding(position, embedding_dim):
        encoding = np.zeros(embedding_dim)
        for i in range(embedding_dim):
            angle = position / math.pow(10000, (2 * (i // 2)) / embedding_dim)
            if i % 2 == 0:
                encoding[i] = math.sin(angle)
            else:
                encoding[i] = math.cos(angle)
        return encoding

    @classmethod
    def position_encoding(cls, sentence, embeddings_dict, d_model):

        final_dict = {}  # Added to store word: final_input mapping
        final_Vector = []
        for pos, word in enumerate(sentence):
            word_vector = np.array(embeddings_dict[word])
            position_vector = cls.get_position_encoding(pos, d_model)
            final_input = word_vector + position_vector
            final_Vector.append(final_input)
            final_dict[word] = final_input.tolist()

            print(f"\nWord: {word}")
            print(f"Embedding:         {word_vector}")
            print(f"Positional Encode: {position_vector}")
            print(f"Final Input:       {final_input}")
        return final_Vector, final_dict
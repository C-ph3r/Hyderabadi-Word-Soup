import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

def cooccurrence_matrix_sentence_generator(preproc_sentences, sentence_cooc=False, window_size=5):
    '''
    Generates a coocurrence matrix
    '''

    co_occurrences = defaultdict(Counter)

    # Compute co-occurrences
    if sentence_cooc == True:
        for sentence in tqdm(preproc_sentences):
            for token_1 in sentence:
                for token_2 in sentence:
                    if token_1 != token_2:
                        co_occurrences[token_1][token_2] += 1
    else:
        for sentence in tqdm(preproc_sentences):
            for i, word in enumerate(sentence):
                for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                    if i != j:
                        co_occurrences[word][sentence[j]] += 1

    #ensure that words are unique
    unique_words = list(set([word for sentence in preproc_sentences for word in sentence]))

    # Initialize the co-occurrence matrix
    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    # Populate the co-occurrence matrix
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    # Create a DataFrame for better readability
    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=1)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=0)

    # Return the co-occurrence matrix
    return co_matrix_df
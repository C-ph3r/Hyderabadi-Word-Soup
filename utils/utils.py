import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'utils')))
import pipeline as p

def cluster_namer(dataset, label_column_name, nr_words=5):
    '''
    Names clusters based on the top tokens that appear on that cluster, using bow.
    args:
        dataset (DataFrame): Dataset with the cluster data to be named
        label_column_name (string): Name of the column where the cluster labels are stored

    returns:
        dataset(DataFrame): Same as the original dataset, but with newly named clusters
    '''
    labels = list(set(dataset[label_column_name]))
    # corpus generator
    corpus = []
    for label in labels:
        label_doc = ""
        for doc in dataset["filtered_review"].loc[dataset[label_column_name] == label]:        
            if isinstance(doc, list):
                # If the document is a list of tokens, join them into a string
                doc = " ".join(doc)
            label_doc = label_doc + " " + doc
        corpus.append(label_doc)
    # Use CountVectorizer for BoW representation
    bow_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")
    label_name_list = []

    for idx, document in enumerate(corpus):
        corpus_bow_td_matrix = bow_vectorizer.fit_transform(corpus)
        corpus_bow_word_list = bow_vectorizer.get_feature_names_out()

        label_vocabulary = p.word_freq_calculator(corpus_bow_td_matrix[idx].toarray(), \
                                                corpus_bow_word_list, df_output=True)
        
        label_vocabulary = label_vocabulary.head(nr_words)
        label_name = ""
        for jdx in range(len(label_vocabulary)):
            label_name = label_name + "_" + label_vocabulary["words"].iloc[jdx]

        label_name_list.append(label_name)

    label_name_dict = dict(zip(labels, label_name_list))
    dataset[label_column_name] = dataset[label_column_name].map(lambda label: label_name_dict[label])

    return dataset
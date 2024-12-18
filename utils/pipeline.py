import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict, Counter
from tqdm import tqdm
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn import metrics
from sklearn.base import BaseEstimator

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



class MainPipeline(BaseEstimator):
    def __init__(self, 
                 print_output = False, 
                 no_emojis = True, 
                 no_hashtags = True,
                 hashtag_retain_words = True,
                 no_newlines = True,
                 no_urls = True,
                 no_punctuation = True,
                 no_stopwords = True,
                 custom_stopwords = [],
                 convert_diacritics = True, 
                 lowercase = True, 
                 lemmatized = True,
                 list_pos = ["n","v","a","r","s"],
                 pos_tags_list = "no_pos",
                 tokenized_output = False):
        
        self.print_output = print_output 
        self.no_emojis = no_emojis
        self.no_hashtags = no_hashtags
        self.hashtag_retain_words = hashtag_retain_words
        self.no_newlines = no_newlines
        self.no_urls = no_urls
        self.no_punctuation = no_punctuation
        self.no_stopwords = no_stopwords
        self.custom_stopwords = custom_stopwords
        self.convert_diacritics = convert_diacritics
        self.lowercase = lowercase
        self.lemmatized = lemmatized
        self.list_pos = list_pos
        self.pos_tags_list = pos_tags_list
        self.tokenized_output = tokenized_output

    def regex_cleaner(self, raw_text):

        #patterns
        newline_pattern = "(\\n)"
        hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
        hashtags_ats_and_word_pattern = "([#@]\w+)"
        emojis_pattern = "([\u2600-\u27FF])"
        url_pattern = "(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?" ##Note that this URL pattern is *even better*
        punctuation_pattern = "[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u007C\u2010-\u2028\ufeff`]+"
        apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
        separated_words_pattern = "(?<=\w\s)([A-Z]\s){2,}"
        ##note that this punctuation_pattern doesn't capture ' this time to allow our tokenizer to separate "don't" into ["do", "n't"]
        
        if self.no_emojis == True:
            clean_text = re.sub(emojis_pattern,"",raw_text)
        else:
            clean_text = raw_text

        if self.no_hashtags == True:
            if self.hashtag_retain_words == True:
                clean_text = re.sub(hashtags_at_pattern,"",clean_text)
            else:
                clean_text = re.sub(hashtags_ats_and_word_pattern,"",clean_text)
            
        if self.no_newlines == True:
            clean_text = re.sub(newline_pattern," ",clean_text)

        if self.no_urls == True:
            clean_text = re.sub(url_pattern,"",clean_text)
        
        if self.no_punctuation == True:
            clean_text = re.sub(punctuation_pattern,"",clean_text)
            clean_text = re.sub(apostrophe_pattern,"",clean_text)

        return clean_text

    def lemmatize_all(self, token):
    
        wordnet_lem = nltk.stem.WordNetLemmatizer()
        for arg_1 in self.list_pos[0]:
            token = wordnet_lem.lemmatize(token, arg_1)
        return token

    def main_pipeline(self, raw_text):
        
        """Preprocess strings according to the parameters"""
        if self.print_output == True:
            print("Preprocessing the following input: \n>> {}".format(raw_text))

        clean_text = self.regex_cleaner(raw_text)

        if self.print_output == True:
            print("Regex cleaner returned the following: \n>> {}".format(clean_text))

        tokenized_text = nltk.tokenize.word_tokenize(clean_text)

        tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
        tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
        tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]

        if self.no_stopwords == True:
            stopwords = nltk.corpus.stopwords.words("english")
            tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]
        
        if self.convert_diacritics == True:
            tokenized_text = [unidecode(token) for token in tokenized_text]

        if self.lemmatized == True:
            tokenized_text = [self.lemmatize_all(token) for token in tokenized_text]
    
        if self.no_stopwords == True:
            tokenized_text = [item for item in tokenized_text if item.lower() not in self.custom_stopwords]

        if self.pos_tags_list == "pos_list" or self.pos_tags_list == "pos_tuples" or self.pos_tags_list == "pos_dictionary":
            pos_tuples = nltk.tag.pos_tag(tokenized_text)
            pos_tags = [pos[1] for pos in pos_tuples]
        
        if self.lowercase == True:
            tokenized_text = [item.lower() for item in tokenized_text]
        
        if self.pos_tags_list == "pos_list":
            return (tokenized_text, pos_tags)
        elif self.pos_tags_list == "pos_tuples":
            return pos_tuples   
        
        else:
            if self.tokenized_output == True:
                return tokenized_text
            else:
                detokenizer = TreebankWordDetokenizer()
                detokens = detokenizer.detokenize(tokenized_text)
                if self.print_output == True:
                    print("Pipeline returning the following result: \n>> {}".format(str(detokens)))
                return str(detokens)


class HermeticClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, preprocessor, vectorizer, classifier, d2v_vector_size=300, d2v_window=6, **kwargs):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.d2v_vector_size = d2v_vector_size
        self.d2v_window = d2v_window

    def fit(self, X, y, **kwargs):

        X_preproc = [self.preprocessor.main_pipeline(doc, **kwargs) for doc in X]

        try:
            X_train = self.vectorizer.fit_transform(X_preproc)
        except AttributeError:
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
            self.d2v_model = self.vectorizer(documents, vector_size=300, window=6, min_count=1, workers=4)
            X_train = [self.d2v_model.dv[idx].tolist() for idx in range(len(X_preproc))]

        y_train = y
            
        try:
            X_train = X_train.toarray()
        except AttributeError:
            pass
        try:
            y_train = y_train.to_numpy()
        except AttributeError:
            pass

        #X_train, y_train = check_X_y(X_train, y_train)

        self.classifier.fit(X_train, y_train)

        self.X_ = X_train
        self.y_ = y_train

        return self

    def predict(self, X_test_raw, **kwargs):

        # Check if fit has been called
        check_is_fitted(self)

        X_test = [self.preprocessor.main_pipeline(doc, **kwargs) for doc in X_test_raw]

        try:
            X_test = self.vectorizer.transform(X_test)
        except AttributeError:
            X_test = [self.d2v_model.infer_vector(word_tokenize(content)).tolist() for content in X_test]

        try:
            X_test = check_array(X_test.toarray())
        except AttributeError:
            X_test = check_array(X_test)
            
        y_pred = self.classifier.predict(X_test)

        return y_pred
    


def regex_cleaner(raw_text, 
            no_emojis = True, 
            no_hashtags = True,
            hashtag_retain_words = True,
            no_newlines = True,
            no_urls = True,
            no_punctuation = True):
    
    #patterns
    newline_pattern = "(\\n)"
    hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
    hashtags_ats_and_word_pattern = "([#@]\w+)"
    emojis_pattern = "([\u2600-\u27FF])"
    url_pattern = "(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?" ##Note that this URL pattern is *even better*
    punctuation_pattern = "[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u007C\u2010-\u2028\ufeff`]+"
    apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
    separated_words_pattern = "(?<=\w\s)([A-Z]\s){2,}"
    ##note that this punctuation_pattern doesn't capture ' this time to allow our tokenizer to separate "don't" into ["do", "n't"]
    
    if no_emojis == True:
        clean_text = re.sub(emojis_pattern,"",raw_text)
    else:
        clean_text = raw_text

    if no_hashtags == True:
        if hashtag_retain_words == True:
            clean_text = re.sub(hashtags_at_pattern,"",clean_text)
        else:
            clean_text = re.sub(hashtags_ats_and_word_pattern,"",clean_text)
        
    if no_newlines == True:
        clean_text = re.sub(newline_pattern," ",clean_text)

    if no_urls == True:
        clean_text = re.sub(url_pattern,"",clean_text)
    
    if no_punctuation == True:
        clean_text = re.sub(punctuation_pattern,"",clean_text)
        clean_text = re.sub(apostrophe_pattern,"",clean_text)

    return clean_text

def lemmatize_all(token, list_pos=["n","v","a","r","s"]):
    
    wordnet_lem = nltk.stem.WordNetLemmatizer()
    for arg_1 in list_pos:
        token = wordnet_lem.lemmatize(token, arg_1)
    return token

def main_pipeline(raw_text, 
                  print_output = True, 
                  no_stopwords = True,
                  custom_stopwords = [],
                  convert_diacritics = True, 
                  lowercase = True, 
                  lemmatized = True,
                  list_pos = ["n","v","a","r","s"],
                  stemmed = False, 
                  pos_tags_list = "no_pos",
                  tokenized_output = False,
                  **kwargs):
    
    """Preprocess strings according to the parameters"""

    clean_text = regex_cleaner(raw_text, **kwargs)
    tokenized_text = nltk.tokenize.word_tokenize(clean_text)

    tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
    tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
    tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]

    if no_stopwords == True:
        stopwords = nltk.corpus.stopwords.words("english")
        tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]
    
    if convert_diacritics == True:
        tokenized_text = [unidecode(token) for token in tokenized_text]

    if lemmatized == True:
        tokenized_text = [lemmatize_all(token, list_pos=list_pos) for token in tokenized_text]
    
    if stemmed == True:
        porterstemmer = nltk.stem.PorterStemmer()
        tokenized_text = [porterstemmer.stem(token) for token in tokenized_text]
 
    if no_stopwords == True:
        tokenized_text = [item for item in tokenized_text if item.lower() not in custom_stopwords]

    if pos_tags_list == "pos_list" or pos_tags_list == "pos_tuples" or pos_tags_list == "pos_dictionary":
        pos_tuples = nltk.tag.pos_tag(tokenized_text)
        pos_tags = [pos[1] for pos in pos_tuples]
    
    if lowercase == True:
        tokenized_text = [item.lower() for item in tokenized_text]

    if print_output == True:
        print(raw_text)
        print(tokenized_text)
    
    if pos_tags_list == "pos_list":
        return (tokenized_text, pos_tags)
    elif pos_tags_list == "pos_tuples":
        return pos_tuples   
    
    else:
        if tokenized_output == True:
            return tokenized_text
        else:
            detokenizer = TreebankWordDetokenizer()
            detokens = detokenizer.detokenize(tokenized_text)
            return str(detokens)      

def cooccurrence_matrix_sentence_generator(preproc_sentences, sentence_cooc=False, window_size=5):

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

def word_freq_calculator(td_matrix, word_list, df_output=True):
    word_counts = np.sum(td_matrix, axis=0).tolist()
    if df_output == False:
        word_counts_dict = dict(zip(word_list, word_counts))
        return word_counts_dict
    else:
        word_counts_df = pd.DataFrame({"words":word_list, "frequency":word_counts})
        word_counts_df = word_counts_df.sort_values(by=["frequency"], ascending=False)
        return word_counts_df

def fold_score_calculator(y_pred, y_test, verbose=False):
    
    # Compute the binary classification scores (accuracy, precision, recall, F1, AUC) for the fold.
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average="weighted")
    recall = metrics.recall_score(y_test, y_pred, average="weighted")
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")

    if verbose == True:
        print("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(acc,prec,recall,f1))
    return (acc, prec, recall, f1)
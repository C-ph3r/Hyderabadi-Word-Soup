import re
import nltk
import pandas as pd
import numpy as np

from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordDetokenizer


def regex_cleaner(raw_text, 
            no_emojis = True, 
            no_hashtags = True,
            hashtag_retain_words = True,
            no_newlines = True,
            no_urls = True,
            no_punctuation = True):
    '''
    Preprocessing steps that require regular expressions
    '''
    # Patterns --------------------------------------------------------------------------
    newline_pattern = "(\\n)"
    hashtags_at_pattern = "([#\@@\u0040\uFF20\uFE6B])"
    hashtags_ats_and_word_pattern = "([#@]\w+)"
    emojis_pattern = "([\u2600-\u27FF])"
    url_pattern = "(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?" ##Note that this URL pattern is *even better*
    punctuation_pattern = "[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+"
    apostrophe_pattern = "'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
    separated_words_pattern = "(?<=\w\s)([A-Z]\s){2,}"
    ##note that this punctuation_pattern doesn't capture ' this time to allow our tokenizer to separate "don't" into ["do", "n't"]
    
    # Emojis ---------------------------------------------------------------------
    if no_emojis == True:
        clean_text = re.sub(emojis_pattern,"",raw_text)
    else:
        clean_text = raw_text


    # Hashtags -------------------------------------------------------------------
    if no_hashtags == True:
        if hashtag_retain_words == True:
            clean_text = re.sub(hashtags_at_pattern,"",clean_text)
        else:
            clean_text = re.sub(hashtags_ats_and_word_pattern,"",clean_text)
    
    # New lines -------------------------------------------------------------------
    if no_newlines == True:
        clean_text = re.sub(newline_pattern," ",clean_text)

    # URLs -------------------------------------------------------------------------
    if no_urls == True:
        clean_text = re.sub(url_pattern,"",clean_text)
    
    # Punctuation ---------------------------------------------------------------------
    if no_punctuation == True:
        clean_text = re.sub(punctuation_pattern,"",clean_text)
        clean_text = re.sub(apostrophe_pattern,"",clean_text)

    return clean_text


def lemmatize_all(token, list_pos=["n","v","a","r","s"]):
    '''
    Apply WordNetLemmatizer to a token
    '''
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
    '''
    Preprocess strings according to the parameters
    '''
    
    # Applying the regex steps
    clean_text = regex_cleaner(raw_text, **kwargs)

    # Applying tokenization
    tokenized_text = nltk.tokenize.word_tokenize(clean_text)
    tokenized_text = [re.sub("'m","am",token) for token in tokenized_text]
    tokenized_text = [re.sub("n't","not",token) for token in tokenized_text]
    tokenized_text = [re.sub("'s","is",token) for token in tokenized_text]

    # Stopwords ---------------------------------------------------------------------------------
    if no_stopwords == True:
        stopwords = nltk.corpus.stopwords.words("english")
        tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords]
    
    # Diacritics ---------------------------------------------------------------------------------
    if convert_diacritics == True:
        tokenized_text = [unidecode(token) for token in tokenized_text]

    # Lemmatization ------------------------------------------------------------------------------
    if lemmatized == True:
        tokenized_text = [lemmatize_all(token, list_pos=list_pos) for token in tokenized_text]
    
    # Stemming ------------------------------------------------------------------------------------
    if stemmed == True:
        porterstemmer = nltk.stem.PorterStemmer()
        tokenized_text = [porterstemmer.stem(token) for token in tokenized_text]
    
    # Stopwords ------------------------------------------------------------------------------------
    if no_stopwords == True:
        tokenized_text = [item for item in tokenized_text if item.lower() not in custom_stopwords]

    # Parts of Speech ------------------------------------------------------------------------------
    if pos_tags_list == "pos_list" or pos_tags_list == "pos_tuples" or pos_tags_list == "pos_dictionary":
        pos_tuples = nltk.tag.pos_tag(tokenized_text)
        pos_tags = [pos[1] for pos in pos_tuples]
    
    # Lowercase -------------------------------------------------------------------
    if lowercase == True:
        tokenized_text = [item.lower() for item in tokenized_text]

    # Verbose -----------------------------------
    if print_output == True:
        print(raw_text)
        print(tokenized_text)
    
    # Return Parts of Speech if they were recorded
    if pos_tags_list == "pos_list":
        return (tokenized_text, pos_tags)
    elif pos_tags_list == "pos_tuples":
        return pos_tuples   
    
    else:
        # Returns the tokenized output
        if tokenized_output == True:
            return tokenized_text
        else:
            # Detokenizes the text
            detokenizer = TreebankWordDetokenizer()
            detokens = detokenizer.detokenize(tokenized_text)
            return str(detokens)
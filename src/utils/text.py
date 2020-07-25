import re

from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from typing import List, Dict
from collections import Counter
from math import log

def clean_text(text: str, stopwords: List[str]) -> List[str]:
    text = re.sub(r"[\"\(\)]", ' ', text).lower()
    text = re.sub(r"[\-\_]", '', text) 
    lem  = WordNetLemmatizer()
    if not isinstance(stopwords, set):
        stopwords = set(stopwords)
    return [lem.lemmatize(w) for w in word_tokenize(text) if (w not in stopwords and not re.match(r"^.*\W.*$", w))]

def calculate_tfidf(text: List[str], ndocs: int=1) -> Dict[str, float]:
    count = Counter(text)
    length = len(text)
    # Calculate tf
    tfidf = {k : v / length for k, v in count.items()}
    # Calculate idf 
    for k, v in tfidf.items():
	    idf = log(ndocs / count[k])
	    tfidf[k] *= idf
    return tfidf

def most_common_ngrams(text, n):
    ngrams = list(nltk.ngrams(text, n))
    counts = Counter(ngrams)
    return sorted(set(ngrams), key=counts.get, reverse=True)

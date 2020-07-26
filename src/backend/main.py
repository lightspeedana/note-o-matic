#!/usr/bin/env python3
import nltk
import gensim.models

import numpy as np

import utils.fetch as fetch
import utils.text as text
import utils.models as models

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import random
import time
import matplotlib.pyplot as plt

# URL = "https://en.wikipedia.org/wiki/United_States"
URL = "https://en.wikipedia.org/wiki/Gerald_B._Greenberg"

title, paragraphs = fetch.parse_webpage(URL)
stopwords = nltk.corpus.stopwords.words('english')
cleaned = text.clean_text(paragraphs, stopwords)
sent_tok = nltk.sent_tokenize(paragraphs)
unique = list(set(cleaned)) 

# text.generate_wordcloud(' '.join(cleaned), stopwords)

model, db, clusters, word_clusters, n_clusters, n_noise = models.word2vec_model(unique, min_count=1, window=5, verbose=True)
# word_clusters = [r for _ in sent_tok for r in _]

print('-'*40)
print(clusters.shape, len(unique))
print(*word_clusters, sep='\n')
print('-'*40)


sent_ting = []

for sent in sent_tok:
    for w in word_clusters:
        for w2 in w:
            if w2 in sent.lower():
                sent_ting.append(sent)

print(*sent_ting, sep='\n')



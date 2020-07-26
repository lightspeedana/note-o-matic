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
unique = list(set(cleaned)) # cast to list so stuff is in the same place, just in case stuff does a fucky wucky otherwise

# text.generate_wordcloud(' '.join(cleaned), stopwords)

model, db, clusters, n_clusters, n_noise = models.word2vec_model(unique, min_count=1, window=5, verbose=True)

print('-'*40)
print(clusters.shape, len(unique))

word_clusters = [np.array(unique)[clusters == i] for i in range(n_clusters)]

print(*word_clusters, sep='\n')

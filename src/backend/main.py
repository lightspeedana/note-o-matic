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
unique = set(cleaned)

# text.generate_wordcloud(' '.join(cleaned), stopwords)

model, db = models.word2vec_model(unique, min_count=1, window=5, verbose=True)
labels = db.labels_
vectors = model.wv.vectors

d = {vectors[i]:labels[i] for i in range(len(labels))}

print('-'*40)
print(g)




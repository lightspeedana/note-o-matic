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

model = gensim.models.Word2Vec([unique], min_count=1, size=len(unique), window=5)

neighbors = NearestNeighbors(n_neighbors=2)
nbrs = neighbors.fit(model.wv.vectors)
t = time.time()
distances, _ = nbrs.kneighbors(model.wv.vectors)
distances = np.sort(distances[:,1], axis=0)
epsilon = np.average(distances[:len(distances)//2])

t = time.time()
clustering = DBSCAN(eps=epsilon, min_samples=5).fit(model.wv.vectors)
print(f"Time taken to DBSCAN : {time.time() - t}")

n_clusters_ = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
n_noise_ = list(clustering.labels_).count(-1)

print(f"Clusters: {n_clusters_} | Noise: {n_noise_}")



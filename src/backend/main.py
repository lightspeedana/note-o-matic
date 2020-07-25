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
distances, indices = nbrs.kneighbors(model.wv.vectors)
distances = distances[:,1]
distances = np.sort(distances, axis=0)

grads = np.gradient(distances)
#grads = np.sort(grads, axis=0)
print(f"Time taken to fit : {time.time() - t}")
print(grads)
plt.plot(distances, label="Distances")
plt.figure()
plt.plot(grads, label="grads")
plt.show()


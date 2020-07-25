#!/usr/bin/env python3
import nltk
import gensim.models

import numpy as np

import utils.fetch as fetch
import utils.text as text

URL = "https://en.wikipedia.org/wiki/United_States"

title, paragraphs = fetch.parse_webpage(URL)
stopwords = nltk.corpus.stopwords.words('english')
cleaned = text.clean_text(paragraphs, stopwords)
# text.generate_wordcloud(' '.join(cleaned), stopwords)

model1 = gensim.models.Word2Vec(cleaned, min_count=1, size=100, window=5)
wvTing = model1.wv



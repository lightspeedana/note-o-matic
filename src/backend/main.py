#!/usr/bin/env python3
import nltk
import gensim.models

import numpy as np

import utils.fetch as fetch
import utils.text as text
import utils.models as models


URL = "https://en.wikipedia.org/wiki/United_States"

title, paragraphs = fetch.parse_webpage(URL)
stopwords = nltk.corpus.stopwords.words('english')
cleaned = text.clean_text(paragraphs, stopwords)
unique = set(cleaned)
# text.generate_wordcloud(' '.join(cleaned), stopwords)

model = gensim.models.Word2Vec([unique], min_count=1, size=len(unique), window=5)







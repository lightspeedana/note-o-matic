#!/usr/bin/env python3
import nltk

import utils.fetch as fetch
import utils.text as text

URL = "https://en.wikipedia.org/wiki/United_States"

title, paragraphs = fetch.parse_webpage(URL)
stopwords = nltk.corpus.stopwords.words('english')
cleaned = text.clean_text(paragraphs, stopwords)

print(cleaned)

text.generate_wordcloud(' '.join(cleaned), stopwords)

#!/usr/bin/env python
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union, Dict
from utils.fetch import *
from utils.text import *

URL = "https://en.wikipedia.org/wiki/United_States"

title, text =  parse_webpage(URL)
stopwords = nltk.corpus.stopwords.words('english')
cleaned = clean_text(text, stopwords)

tdidf = calculate_tfidf(cleaned)

print(tdidf)



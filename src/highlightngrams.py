import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from utils.fetch import *
from utils.text import *
from collections import Counter

URL = "https://en.wikipedia.org/wiki/United_States"

title, text = parse_webpage(URL)

stopwords = nltk.corpus.stopwords.words('english')
cleaned = clean_text(text, stopwords)

twograms = most_common_ngrams(cleaned, 2)
threegrams = most_common_ngrams(cleaned, 3)

print(twograms[:15])
print(threegrams[:15])

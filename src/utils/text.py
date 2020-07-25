import re

import matplotlib.pyplot as plt

from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
from typing import List, Dict
from collections import Counter
from math import log


def clean_text(text: str, stopwords: List[str]) -> List[str]:
    text = re.sub(r"[\"\(\)]", ' ', text).lower()
    text = re.sub(r"[\-\_]", '', text) 
    lem  = WordNetLemmatizer()
    return [lem.lemmatize(w) for w in word_tokenize(text) if (w not in stopwords and not re.match(r"^.*\W.*$", w))]

def calculate_tfidf(text: List[str], ndocs: int=1) -> Dict[str, float]:
    count = Counter(text)
    length = len(text)
    # Calculate tf
    tfidf = {k : v / length for k, v in count.items()}
    # Calculate idf 
    for k, v in tfidf.items():
	    idf = log(ndocs / count[k])
	    tfidf[k] *= idf
    return tfidf

def generate_wordcloud(text: str, stopwords: List[str]) -> None:
    def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
        h = int(360.0 * 21.0 / 255.0)
        s = int(100.0 * 255.0 / 255.0)
        l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

        return "hsl({}, {}%, {}%)".format(h, s, l)

    wordcloud = WordCloud(background_color='white',
                            stopwords=stopwords,
                            max_words=100,
                            width=800, 
                            height=600,
			    color_func=random_color_func,
                            max_font_size=150).generate(str(text))
    fig = plt.figure(1, figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis('off')
    # plt.show()
    plt.savefig('wordcloud.png', bbox_inches='tight')

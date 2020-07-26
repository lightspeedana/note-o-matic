import re

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import nltk
import nltk.stem.wordnet as wordnet
import wordcloud as wc
import math
import io
import base64

from collections import Counter
from typing import List, Dict


def clean_text(text: str, stopwords: List[str]) -> List[str]:
    text = re.sub(r"[\"\(\)]", " ", text).lower()
    text = re.sub(r"[\-\_]", "", text)
    lem = wordnet.WordNetLemmatizer()
    if not isinstance(stopwords, set):
        stopwords = set(stopwords)
    return [
        lem.lemmatize(w)
        for w in nltk.word_tokenize(text)
        if (w not in stopwords and not re.match(r"^.*[^a-zA-Z].*$", w))
    ]


def calculate_tfidf(text: List[str], ndocs: int = 1) -> Dict[str, float]:
    count = Counter(text)
    length = len(text)
    # Calculate tf
    tfidf = {k: v / length for k, v in count.items()}
    # Calculate idf
    for k, v in tfidf.items():
        idf = math.log(ndocs / count[k])
        tfidf[k] *= idf
    return tfidf


def generate_wordcloud(text: str, stopwords: List[str]) -> None:
    def random_color_func(
        word=None,
        font_size=None,
        position=None,
        orientation=None,
        font_path=None,
        random_state=None,
    ):
        h = int(360.0 * 21.0 / 255.0)
        s = int(100.0 * 255.0 / 255.0)
        l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

        return "hsl({}, {}%, {}%)".format(h, s, l)

    wordcloud = wc.WordCloud(
        background_color="white",
        stopwords=stopwords,
        max_words=100,
        width=800,
        height=600,
        color_func=random_color_func,
        max_font_size=150,
    ).generate(str(text))
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    figfile = io.BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    return base64.b64encode(figfile.getvalue()).decode('utf-8')
    
    # plt.show()
    # plt.savefig("wordcloud.png", bbox_inches="tight")

def most_common_ngrams(text: str, n: int) -> List[list]:
    ngrams = list(nltk.ngrams(text, n))
    counts = Counter(ngrams)
    return sorted(set(ngrams), key=counts.get, reverse=True)


def create_notes(paragraphs: str, word_clusters: List[List[str]]) -> str:
    word_clusters = set().union(*word_clusters)
    notes = [
        sent for sent in nltk.sent_tokenize(paragraphs)
        if any(w in word_clusters for w in sent.split())
        ]
    return '\n'.join(notes)


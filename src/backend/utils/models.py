import gensim
import numpy as np

from typing import List

def word2vec_model(unique_text: List[str], *args):
    model = gensim.models.Word2Vec([unique_text], *args)


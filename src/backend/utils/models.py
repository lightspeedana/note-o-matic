import gensim
import numpy as np
import sklearn.cluster
import sklearn.neighbors

from typing import List

def word2vec_model(unique_text: set, min_count: int=1, window: int=5, n_neighbors: int=2, min_samples: int=4, verbose: bool=False):
    """
    Returns a word2vec model, a dbscan, clusters, the number of clusters, and the amount of noise
    """
    model = gensim.models.Word2Vec([unique_text], min_count=min_count, size=len(unique_text), window=window)
    vec = model.wv.vectors
    
    # Get optimal epsilon
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neighbors.fit(vec)
    distances, _ = nbrs.kneighbors(vec)
    distances = np.sort(distances[:,1], axis=0)
    epsilon = np.average(distances[:len(distances)//2])
    
    # Run cluster detection ting
    db = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=min_samples).fit(vec)
    clusters = db.labels_
    n_clusters = len(set(clusters)) - (-1 in clusters)
    n_noise = list(clusters).count(-1)
    word_clusters = [np.array(unique_text)[clusters == i] for i in range(n_clusters)]
    if verbose:
        print(f"Clusters: {n_clusters} | Noise: {n_noise}")
    
    return model, db, clusters, word_clusters, n_clusters, n_noise

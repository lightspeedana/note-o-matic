import gensim
import numpy as np
import sklearn.cluster
import sklearn.neighbors


from typing import List

def word2vec_model(unique_text: List[str], verbose: bool=False, *args):
    model = gensim.models.Word2Vec([unique_text], *args)
    vec = model.wv.vectors
    
    # Get optimal epslion
    nbrs = sklearn.neighbors.fit(vec)
    distances, _ = nbrs.kneighbors(vec)
    distances = np.sort(distances[:,1], axis=0)
    epsilon = np.average(distances[:len(distances)//2])
    
    db = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=5).fit(vec)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = list(db.labels_).count(-1)
    if verbose:
        print(f"Clusters: {n_clusters_} | Noise: {n_noise_}")



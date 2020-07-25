import gensim
import numpy as np
import sklearn.cluster
import sklearn.neighbors


from typing import List

def word2vec_model(unique_text: set, min_count: int=1, window: int=5, n_neighbors: int=2, min_samples: int=5, verbose: bool=False) -> None: 
    model = gensim.models.Word2Vec([unique_text], min_count=min_count, size=len(unique_text), window=window)
    vec = model.wv.vectors
    
    # Get optimal epslion
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neighbors.fit(vec)
    distances, _ = nbrs.kneighbors(vec)
    distances = np.sort(distances[:,1], axis=0)
    epsilon = np.average(distances[:len(distances)//2])
    
    # Run cluster detection ting
    db = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=min_samples).fit(vec)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = list(db.labels_).count(-1)
    if verbose:
        print(f"Clusters: {n_clusters} | Noise: {n_noise}")
    
    return model, db


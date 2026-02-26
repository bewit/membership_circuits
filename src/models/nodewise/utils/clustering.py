import timeit
import torch
import numpy as np
from sklearn.cluster import KMeans as SKMeans


def kmeans_sklearn(data: np.ndarray, n_clusters: int = 2, n_init: int = 5, max_iter: int = 50, seed: int = 47, **kwargs):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    km = SKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=seed)
    cluster_assignments = km.fit_predict(data)
    return cluster_assignments


def kmeans_faiss(data: np.ndarray, n_clusters: int = 2, n_init: int = 5, max_iter: int = 50, seed: int = 47, **kwargs):
    import faiss
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    km = faiss.Kmeans(d=data.shape[1], k=n_clusters, niter=max_iter, nredo=n_init, gpu=False, seed=seed)
    km.cp.min_points_per_centroid = 1
    km.train(data)
    D, I = km.index.search(data, 1)
    cluster_assignments = I.ravel()
    return cluster_assignments



def _timing(fct, args, n=100):
    t = 0
    for i in range(n):
        start_time = timeit.default_timer()
        fct(*args)
        time_diff = timeit.default_timer() - start_time
        t += time_diff
    return t



if __name__ == "__main__":
    data = torch.rand((10000, 100))
    duration_faiss = _timing(kmeans_faiss, args=(data,), n=100)
    duration_sklearn = _timing(kmeans_sklearn, args=(data,), n=100)
    print(duration_faiss)
    print(duration_sklearn)
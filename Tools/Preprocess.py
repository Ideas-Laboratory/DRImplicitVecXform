import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import euclidean_distances


def normalize(x, low=-1, up=1):
    data_shape = x.shape
    n = data_shape[0]
    dim = data_shape[1]
    new_x = np.zeros(data_shape)
    min_v = np.zeros((1, dim))
    max_v = np.zeros((1, dim))

    for i in range(0, dim):
        min_v[0, i] = min(x[:, i])
        max_v[0, i] = max(x[:, i])
    for i in range(0, n):
        for j in range(0, dim):
            if min_v[0, j] == max_v[0, j]:
                new_x[i, j] = 0
                continue
            new_x[i, j] = (x[i, j]-min_v[0, j])/(max_v[0, j]-min_v[0, j])*(up-low)+low

    return new_x


def knn(data, k):
    nbr_s = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distance, index = nbr_s.kneighbors(data)
    return index


def has_repeat(X):
    (n, m) = X.shape
    D = euclidean_distances(X)

    repeat = False
    repeat_index = []
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            else:
                temp_bool = True
                for k in range(0, m):
                    if X[i, k] != X[j, k]:
                        temp_bool = False
                if temp_bool:
                    temp_number = max(i, j)
                    if not temp_number in repeat_index:
                        repeat_index.append(max(i, j))
                        repeat = True

    return repeat


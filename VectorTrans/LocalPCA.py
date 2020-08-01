from sklearn.decomposition import PCA
import numpy as np


def local_pca_dn(data):
    (n, m) = data.shape
    if n <= m:
        mean_data = np.mean(data, axis=0)
        X = data - mean_data
        C = np.matmul(np.transpose(X), X) / n
        eigen_values, eigen_vectors = np.linalg.eig(C)
        eig_idx = np.argpartition(eigen_values, -m)[-m:]
        eig_idx = eig_idx[np.argsort(-eigen_values[eig_idx])]
        vectors = eigen_vectors[:, eig_idx]
        vectors = np.transpose(vectors)
        values = eigen_values[eig_idx]
    else:
        data_shape = data.shape
        local_pca = PCA(n_components=data_shape[1], copy=True, whiten=True)
        local_pca.fit(data)
        vectors = local_pca.components_
        values = local_pca.explained_variance_
    return vectors, values

import numpy as np
from VectorTrans import LocalPCA
from VectorTrans import Metric
from VectorTrans.DRTrans import DRTrans
from Tools import Preprocess
from sklearn.decomposition import PCA


class PCATrans(DRTrans):
    def __init__(self, X, label=None):
        super().__init__()
        self.X = X
        if label is None:
            self.label = np.ones((X.shape[0], 1))
        else:
            self.label = label
        self.n_samples = X.shape[0]
        self.Y = None
        self.y_add_list = []
        self.y_sub_list = []
        self.derivative = None
        self.point_error = None

        self.k = 0
        self.eigen_number = 2

    def transeform_(self, eigen_weights, eigen_vectors_list, yita):
        y_add_list = []
        y_sub_list = []
        (n, dim) = self.X.shape
        for loop_index in range(0, self.eigen_number):
            eigenvectors = eigen_vectors_list[loop_index]
            x_add_v = np.zeros((n, dim))
            x_sub_v = np.zeros((n, dim))
            for i in range(0, n):
                x_add_v[i, :] = self.X[i, :] + yita * eigen_weights[i, loop_index] * eigenvectors[i, :]
                x_sub_v[i, :] = self.X[i, :] - yita * eigen_weights[i, loop_index] * eigenvectors[i, :]

            y_add_v = np.matmul(x_add_v, self.derivative)
            y_sub_v = np.matmul(x_sub_v, self.derivative)
            y_add_list.append(y_add_v)
            y_sub_list.append(y_sub_v)

        return y_add_list, y_sub_list

    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.1):
        (n, dim) = self.X.shape
        if MAX_EIGEN_COUNT > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            MAX_EIGEN_COUNT = dim

        self.k = nbrs_k
        self.eigen_number = MAX_EIGEN_COUNT

        knn = Preprocess.knn(self.X, nbrs_k)

        eigen_vectors_list = []
        eigen_values = np.zeros((n, dim))
        eigen_weights = np.ones((n, dim))

        for i in range(0, MAX_EIGEN_COUNT):
            eigen_vectors_list.append(np.zeros((n, dim)))

        for i in range(0, n):
            local_data = np.zeros((nbrs_k, dim))
            for j in range(0, nbrs_k):
                local_data[j, :] = self.X[knn[i, j], :]
            temp_vectors, eigen_values[i, :] = LocalPCA.local_pca_dn(local_data)

            for j in range(0, MAX_EIGEN_COUNT):
                eigenvectors = eigen_vectors_list[j]
                eigenvectors[i, :] = temp_vectors[j, :]

            temp_eigen_sum = sum(eigen_values[i, :])
            for j in range(0, dim):
                eigen_weights[i, j] = eigen_values[i, j] / temp_eigen_sum

        print("Computing dimension reduction...")
        pca = PCA(n_components=2, copy=True, whiten=True)
        pca.fit(self.X)
        P_ = pca.components_
        P = np.transpose(P_)
        self.Y = np.matmul(self.X, P)
        self.derivative = P
        self.point_error = Metric.pca_error(self.X, self.Y)
        self.linearity = Metric.linearity(eigen_values)
        self.y_add_list, self.y_sub_list = self.transeform_(eigen_weights, eigen_vectors_list, yita)
        return self.Y, self.y_add_list, self.y_sub_list





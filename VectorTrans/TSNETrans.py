import numpy as np
from DR import TSNE
from Tools import Preprocess
from VectorTrans import LocalPCA
from VectorTrans.VectorTransform import VectorTransform
from VectorTrans.DRTrans import DRTrans
from Derivative.TSNEDerivative import TSNEDerivative
from VectorTrans import Metric


class TSNETrans(DRTrans):
    def __init__(self, X, label=None, y_init=None, perplexity=30.0):
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
        self.Px0 = None  #
        self.Px = None  #
        self.Q = None  #
        self.beta = None  #
        self.derivative = None
        self.point_error = None
        self.init_y(y_init, perplexity=perplexity)
        self.k = 0
        self.eigen_number = 2

    def init_y(self, Y0, perplexity=30.0):
        print("Computing dimension reduction...")
        t_sne = TSNE.TSNE(n_component=2, perplexity=perplexity)
        if Y0 is None:
            Y = t_sne.fit_transform(self.X, max_iter=20000)
        else:
            Y = t_sne.fit_transform(self.X, max_iter=1000, early_exaggerate=False, y_init=Y0)
        self.Y = Y
        self.beta = t_sne.beta
        self.Px0 = t_sne.P0
        self.Px = t_sne.P
        self.Q = t_sne.Q

    def transform_(self, vectors_list, weights):
        print("Computing derivative...")
        derivative = TSNEDerivative()
        self.derivative = derivative.get_derivative(self.X, self.Y, self.Px, self.Q, self.Px0, self.beta)
        vector_transform = VectorTransform(self.Y, self.derivative)
        self.y_add_list, self.y_sub_list = vector_transform.transform_all(vectors_list, weights)

        return self.y_add_list, self.y_sub_list

    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.1):
        print("Transform t-SNE...")
        (n, dim) = self.X.shape

        if MAX_EIGEN_COUNT > dim:
            print("Warning: the input MAX_EIGEN_COUNT is too large.")
            MAX_EIGEN_COUNT = dim

        self.k = nbrs_k
        self.eigen_number = MAX_EIGEN_COUNT

        # k neighbors
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

        y_add_list, y_sub_list = self.transform_(eigen_vectors_list, yita*eigen_weights)

        self.point_error = Metric.mds_stress(self.Px, self.Q)  #
        self.linearity = Metric.linearity(eigen_values)

        return self.Y, y_add_list, y_sub_list

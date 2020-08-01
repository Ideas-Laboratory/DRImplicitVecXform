import numpy as np


class VectorTransformPlus:
    def __init__(self, Y, P):
        self.Y = Y
        self.P = P
        self.y_add_list = []
        self.y_sub_list = []
        (self.n_samples, m) = Y.shape

    def transform(self, vectors):
        (n, d) = vectors.shape
        Y2 = np.zeros((n, 2))

        for i in range(0, n):
            Y2[i, 0] = np.inner(self.P[i*2, :], vectors[i, :])
            Y2[i, 1] = np.inner(self.P[i*2+1, :], vectors[i, :])

        Y2 = Y2 + self.Y

        return Y2

    def transform_all(self, vector_list, weights):
        eigen_number = len(vector_list)
        self.y_add_list = []
        self.y_sub_list = []

        for loop_index in range(0, eigen_number):
            vectors = vector_list[loop_index].copy()
            (n, d) = vectors.shape
            for i in range(0, n):
                vectors[i, :] = vectors[i, :] * weights[i, loop_index]
            y_add = self.transform(vectors)
            y_sub = self.transform(-1*vectors)
            self.y_add_list.append(y_add)
            self.y_sub_list.append(y_sub)

        return self.y_add_list, self.y_sub_list


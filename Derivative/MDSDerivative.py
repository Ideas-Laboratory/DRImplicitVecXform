import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import math
import time

# The derivative of MDS using Stress Model


def hessian_y_matrix(Dx, Dy, Y):
    begin_time = time.time()
    (n, m) = Y.shape
    H = np.zeros((n*m, n*m))
    Dy2 = Dy.copy()
    Dy2[range(n), range(n)] = 1.0

    for a in range(0, n):  # n
        dY = np.tile(Y[a, :], (n, 1)) - Y
        W = np.tile(2 * Dx[a, :] / (Dy2[a, :] ** 3), (m, 1)).T
        for c in range(0, n):
            H_sub = np.zeros((m, m))
            if a == c:
                H_sub = np.matmul(dY.T, W * dY)
                dH = np.eye(m)*(2*n-2-2*np.sum(Dx[a, :]/Dy2[a, :]))
                H_sub = H_sub + dH
            else:
                left_sub = (-2+2*Dx[a, c]/Dy2[a, c]) * np.eye(m)
                right_sub = W[c, 0] * np.outer(dY[c, :], dY[c, :])
                H_sub = left_sub - right_sub
            H[a*m:a*m+m, c*m:c*m+m] = H_sub[:, :]
        if a % 100 == 0:
            print("\tComputing Hessian matrix for  %d of %d..." % (a, n))
    finish_time = time.time()
    print("\tHessian matrix has been computed, the total time is %d s" % (finish_time-begin_time))
    return H


def derivative_X_matrix(Dx, Dy, X, Y):
    begin_time = time.time()
    (n, d) = X.shape
    (n2, m) = Y.shape
    Dx2 = Dx.copy()
    Dy2 = Dy.copy()
    Dx2[range(n), range(n)] = 1.0
    Dy2[range(n), range(n)] = 1.0

    J = np.zeros((n * m, n * d))
    for a in range(0, n):
        Wy = np.tile(1.0 / Dy2[a, :], (m, 1)).T
        Wx = np.tile(1.0 / Dx2[a, :], (d, 1)).T
        dY = np.tile(Y[a, :], (n, 1)) - Y
        dX = np.tile(X[a, :], (n, 1)) - X
        for b in range(0, n):
            H_sub = np.zeros((m, d))
            if a == b:
                H_sub = -2 * np.matmul((Wy * dY).T, Wx * dX)
            else:
                H_sub = 2 * Wy[b, 0] * Wx[b, 0] * np.outer(dY[b, :], dX[b, :])
            J[a*m:a*m+m, b*d:b*d+d] = H_sub[:, :]
        if a % 100 == 0:
            print("\tComputing matrix for  %d of %d..." % (a, n))
    finish_time = time.time()
    print("\tMatrix has been computed, the total time is %d s" % (finish_time-begin_time))

    return J


def Jyx(H, J):
    H_ = np.linalg.pinv(H)  # inv
    P = (-1) * np.matmul(H_, J)
    return P


class MDSDerivative:
    def __init__(self):
        self.H = None
        self.J_yx = None
        self.P = None
        self.Dx = None
        self.Dy = None

    def get_derivative(self, X, Y):
        Dx = euclidean_distances(X)
        Dy = euclidean_distances(Y)
        self.Dx = Dx
        self.Dy = Dy
        self.H = hessian_y_matrix(Dx, Dy, Y)
        self.J_yx = derivative_X_matrix(Dx, Dy, X, Y)
        self.P = Jyx(self.H, self.J_yx)

        return self.P


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import math


def hessian_y_matrix(Dx, Dy, Y):
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

    return H


def Jyx_Plus_memory(Dx, Dy, X, Y, H):
    (n, d) = X.shape
    (n2, m) = Y.shape
    P = np.zeros((n*2, d))
    H = np.linalg.pinv(H)
    Dx2 = Dx.copy()
    Dy2 = Dy.copy()
    Dx2[range(n), range(n)] = 1.0
    Dy2[range(n), range(n)] = 1.0

    for b in range(0, n):
        Jb = np.zeros((m*n, d))
        for a in range(0, n):
            Wy = np.tile(1.0 / Dy2[a, :], (m, 1)).T
            Wx = np.tile(1.0 / Dx2[a, :], (d, 1)).T
            dY = np.tile(Y[a, :], (n, 1)) - Y
            dX = np.tile(X[a, :], (n, 1)) - X
            if a == b:
                H_sub = -2 * np.matmul((Wy * dY).T, Wx * dX)
            else:
                H_sub = 2 * Wy[b, 0] * Wx[b, 0] * np.outer(dY[b, :], dX[b, :])
            Jb[a * m:a * m + m, :] = H_sub[:, :]
        P[b*m:b*m+m, :] = np.matmul(H[b*m:b*m+m, :], Jb)
    P = -1 * P
    return P


class MDSDerivativePlus:
    def __init__(self):
        self.H = None
        self.J_yx = None
        self.P = None
        self.Dx = None
        self.Dy = None

    def get_derivative(self, X, Y):
        Dx = euclidean_distances(X)
        Dy = euclidean_distances(Y)
        H = hessian_y_matrix(Dx, Dy, Y)
        self.P = Jyx_Plus_memory(Dx, Dy, X, Y, H)

        return self.P

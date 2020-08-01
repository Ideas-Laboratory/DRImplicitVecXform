import numpy as np
from sklearn.metrics import euclidean_distances
import math
import time


def hessian_y_matrix(Dy, P, Q, Y):
    (n, m) = Y.shape
    H = np.zeros((n*m, n*m))
    PQ = P - Q
    E = 1.0 / (1 + Dy**2)

    for a in range(0, n):  # n
        dY = np.tile(Y[a, :], (n, 1)) - Y
        Wq = np.tile(Q[a, :], (m, 1)).T
        Wd = np.tile(E[a, :], (m, 1)).T
        Wp = np.tile(PQ[a, :], (m, 1)).T
        wY = Wd*dY

        for c in range(0, n):  # n
            H_sub = np.zeros((m, m))
            if a == c:
                H_sub1 = (-2)*np.matmul((Wp * wY).T, wY)  #
                H_sub2 = np.dot(PQ[a, :], E[a, :]) * np.eye(m)  #
                dY_in2 = 4 * np.matmul(Q[a, :], wY)  #
                dY_in = (-2) * wY + dY_in2  #
                H_sub3 = (-1) * np.matmul(wY.T, Wq * dY_in)  #
                H_sub = (H_sub1 + H_sub2 + H_sub3)*4
            else:
                dYc = np.tile(Y[c, :], (n, 1)) - Y
                sub2_1 = np.matmul(E[c, :]**2, dYc)
                sub2_2 = np.matmul(Q[a, :]**2, dY)
                H_sub2 = (-4) * np.outer(sub2_2, sub2_1)  #
                H_sub3 = 2 * PQ[a, c] * (E[a, c]**2) * np.outer(dY[c, :], dY[c, :]) - PQ[a, c]*E[a, c]*np.eye(m)
                H_sub1 = (-2)*Q[a, c]*E[a, c]*E[a, c]*np.outer(dY[c, :], dY[c, :])
                H_sub = (H_sub1 + H_sub2 + H_sub3)*4

            H[a*m:a*m+m, c*m:c*m+m] = H_sub[:, :]
    return H


def derivative_matrix_memory(X, Y, Dy, beta, P0, H):
    (n, dim) = X.shape
    (n_, m) = Y.shape
    Hinv = np.linalg.pinv(H)

    E = 1.0 / (1+Dy**2)
    Wbeta = np.zeros((n, dim))
    for i in range(0, n):
        Wbeta[i, :] = 2*beta[i]

    P = np.zeros((2*n, dim))
    for c in range(0, n):
        JJ = np.zeros((2*n, dim))
        for a in range(0, n):
            dY = np.tile(Y[a, :], (n, 1)) - Y
            Wd = np.tile(E[a, :], (m, 1)).T
            wY = (Wd * dY).T

            dX = np.tile(X[a, :], (n, 1)) - X
            Wp2 = np.tile(P0[:, a], (dim, 1)).T
            Wp1 = np.tile(P0[a, :], (dim, 1)).T

            J_sub = np.zeros((m, dim))
            if a == c:
                M1 = (Wp2 * (Wp2 - 1) * Wbeta) * dX
                M2 = (Wp1 * beta[a] * 2) * (X - np.matmul(P0[a, :], X))
                J_sub = 2 / n * np.matmul(wY, M1 + M2)
            else:
                dXc = np.tile(X[c, :], (n, 1)) - X
                Wp3 = np.tile(P0[:, c], (dim, 1)).T
                M1 = E[a, c] * np.outer(dY[c, :],
                                        P0[c, a] * Wbeta[c, 0] * (X[a, :] - np.matmul(P0[c, :], X)) + P0[a, c] * Wbeta[
                                            a, 0] * dX[c, :])
                M2 = np.matmul(wY, Wp3 * Wp2 * Wbeta * dXc)
                M3 = np.outer(np.matmul(wY, P0[a, :].T), P0[a, c] * Wbeta[a, 0] * dXc[a, :])
                J_sub = (M1 + M2 + M3) * (2 / n)
            JJ[a * m:a * m + m, :] = J_sub[:, :]
        P[c*m:c*m+m, :] = np.matmul(Hinv[c*m:c*m+m, :], JJ)

    P = -1 * P
    return P


class TSNEDerivativePlus:
    def __init__(self):
        self.P = None

    def get_derivative(self, X, Y, P, Q, P0, beta):
        Dy = euclidean_distances(Y)
        print("Hessian...")
        H = hessian_y_matrix(Dy, P, Q, Y)
        self.P = derivative_matrix_memory(X, Y, Dy, beta, P0, H)
        return self.P

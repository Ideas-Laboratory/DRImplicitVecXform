import numpy as np
import matplotlib.pyplot as plt

# This code was implemented by Laurens van der Maaten on 20-12-08.
# We modified some detailed. You can get the original version from
# https://lvdmaaten.github.io/tsne/


class TSNE:
    def __init__(self, n_component=2, perplexity=30.0):
        """
        init function
        :param n_component: int, optional (default: 2). Dimension of the embedded space
        :param perplexity: float, optional (default: 30.0).
                        Larger datasets usually require a larger perplexity. Consider selecting a value
                        between 5 and 50. Different values can result in significanlty different results.
        """
        self.n_component = n_component
        self.perplexity = perplexity
        self.beta = None
        self.kl = []
        self.final_kl = None
        self.final_iter = 0
        self.P = None
        self.P0 = None
        self.Q = None

    def Hbeta(self, D=np.array([]), beta=1.0):
        """
            Compute the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
        """

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sum_p = sum(P)
        H = np.log(sum_p) + beta * np.sum(D * P) / sum_p
        P = P / sum_p
        return H, P

    def x2p(self, X=np.array([]), tol=1e-12, perplexity=30.0):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

        # Initialize some variables
        # print("\tComputing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        beta = np.ones((n, 1)) / np.max(D)  # modified
        logU = np.log(perplexity)

        # Loop over all datapoints
        for i in range(n):

            # Print progress
            # if i % 500 == 0:
            #     print("\tComputing P-values for point %d of %d..." % (i, n))

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 1000:  #

                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        # Return final P-matrix
        # print("\tMean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        self.beta = beta
        return P

    def fit_transform(self, X, max_iter=1000, early_exaggerate=True, y_init=None):
        # print("\tearly-exaggerate: ", early_exaggerate)
        (n, d) = X.shape
        no_dims = self.n_component
        initial_momentum = 0.5
        final_momentum = 0.8
        if not early_exaggerate:
            final_momentum = 0.0
        eta = 500
        min_gain = 0.01

        # Initialize variables
        Y2 = np.random.randn(n, no_dims)
        if y_init is None:
            Y2 = np.random.randn(n, no_dims)
        else:
            Y2 = y_init
        dY = np.zeros((n, no_dims))
        iY = np.zeros((n, no_dims))
        gains = np.ones((n, no_dims))

        # Compute P-values
        P = self.x2p(X, 1e-15, self.perplexity)
        self.P0 = P.copy()
        P = P + np.transpose(P)
        # P = P / np.sum(P)
        P = P / (2*n)
        P = np.maximum(P, 1e-120)
        self.P = P.copy()

        if early_exaggerate:
            P = P * 4.  # early exaggeration

        # Run iterations
        Y = Y2
        for iter in range(max_iter):
            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-120)

            # Compute gradient
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

            # Perform the update
            if iter < 20 and early_exaggerate:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            if not early_exaggerate:
                gains[gains < min_gain] = min_gain
            iY = momentum * iY - (eta / (np.sqrt(0.1*iter+0.1))) * (gains * dY)  # 改了可变步长
            Y = Y + iY

            # Compute current value of cost function
            # if (iter + 1) % 1000 == 0 and show_progress:
            #     C = np.sum(P * np.log(P / Q))
            #     print("\tIteration %d: error is %f" % (iter + 1, C))
            #     # print("eta = ", eta)

            # Stop lying about P-values
            if iter == 100 and early_exaggerate:
                P = P / 4.

        # update the probability in n_componments dimension space
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-120)
        self.Q = Q

        return Y


if __name__ == '__main__':
    print("t-SNE...")
    X = np.loadtxt("..\\Data\\data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt("..\\Data\\label.csv", dtype=np.int, delimiter=",")

    t_sne = TSNE(n_component=2, perplexity=30.0)
    Y = t_sne.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=label)
    plt.show()

    print(X.shape)






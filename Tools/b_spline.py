import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import math


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
        网址 https://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree, 1, degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree, 1, count-1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree, count+degree+degree-1)
    else:
        kv = np.clip(np.arange(count+degree+1)-degree, 0, count-degree)

    # Calculate query range
    u = np.linspace(periodic, (count - degree), n)

    # Calculate result
    return np.array(si.splev(u, (kv, cv.T, degree))).T


def spline_aspect(current):
    n_points = len(current)
    length_list = []
    half = n_points // 2
    for i in range(0, half):
        point1 = current[i]
        point2 = current[i + half]
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        d = math.sqrt(dx * dx + dy * dy)
        length_list.append(d)

    current_length = max(length_list)
    current_width = min(length_list)

    if current_length != 0:
        return current_length / current_width
    else:
        return 0


def test():
    x = [51, 188, 322, 306, 68]
    y = [51, 69, 95, 392, 185]
    for i in range(0, len(y)):
        y[i] = y[i] * -1

    plt.scatter(x, y, marker='o')

    X = np.array([[51, -51],
                  [188, -69],
                  [322, -95],
                  [306, -392],
                  [68, -185]])
    bs = bspline(X, degree=2, periodic=True)
    (n, m) = bs.shape
    plt.plot(bs[:, 0], bs[:, 1], c='deepskyblue')
    plt.plot([bs[0, 0], bs[n-1, 0]], [bs[0, 1], bs[n-1, 1]], c='deepskyblue')

    bs3 = bspline(X, degree=3, periodic=True)
    plt.plot(bs3[:, 0], bs3[:, 1], c='r')
    plt.plot([bs3[0, 0], bs3[n - 1, 0]], [bs3[0, 1], bs3[n - 1, 1]], c='r')

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    test()

import numpy as np
from sklearn.metrics import euclidean_distances
from Tools import Preprocess
import numpy.linalg as LA


def pca_error(X, Y):
    (n, m) = X.shape
    X2 = X - np.mean(X, axis=0)
    Y2 = Y - np.mean(Y, axis=0)

    distance = np.zeros((n, 1))
    for i in range(0, n):
        distance[i] = np.linalg.norm(X2[i, :])**2 - np.linalg.norm(Y2[i, :])**2

    return distance


def mds_stress(X, Y):
    (n, m) = X.shape
    Dx = euclidean_distances(X)
    Dy = euclidean_distances(Y)

    D = 0.5 * (Dx - Dy) ** 2
    stress = np.sum(D, axis=1)

    return stress.reshape((n, 1))


def tsne_kl(P, Q):
    (n, m) = P.shape
    kl = np.zeros((n, 1))

    P2 = np.maximum(P, 1e-12)
    Q2 = np.maximum(Q, 1e-12)

    M = P * np.log(P2/Q2)
    for i in range(0, n):
        kl[i] = np.sum(M[i, :])

    return kl


def angle_v1_v2(y1, y2, y=None):
    data_shape = y1.shape
    n = data_shape[0]
    m = data_shape[1]

    if y is None:
        y = np.zeros(data_shape)
    v1 = y1 - y
    v2 = y2 - y

    angles = np.zeros((n, 1))
    zeros_count = 0
    for i in range(0, n):
        point1 = v1[i, :]
        point2 = v2[i, :]
        dot_value = np.matmul(point1, np.transpose(point2))

        temp_data = (LA.norm(point1) * LA.norm(point2))
        if temp_data == 0:
            angles[i] = 90
            zeros_count = zeros_count + 1
            continue

        cos_value = dot_value / temp_data
        angle = np.arccos(cos_value)
        angle = angle*180/np.pi
        if angle > 90:
            angle = 180 - angle

        angles[i] = angle

    if zeros_count > 1:
        print('[processData.angle_v1_v2]\t '+str(zeros_count)+' have no change')

    return angles


def trustworthniess(X, Y, k):
    (n, m) = X.shape

    Knn_x = Preprocess.knn(X, k)
    Knn_y = Preprocess.knn(Y, k)

    trust = np.zeros((n, 1))
    for i in range(0, n):
        rank = 0
        for j in range(0, k):
            if not (Knn_y[i, j] in Knn_x[i, :]):
                rank = rank + j - k
        trust[i] = rank
    trust = np.ones((n, 1)) - trust * 2 / (k*(2*m-3*k-1))
    return trust


def triangle_area(point_a, point_b, point_c):
    temp1 = point_a[0]*(point_b[1]-point_c[1])
    temp2 = point_b[0]*(point_c[1]-point_a[1])
    temp3 = point_c[0]*(point_a[1]-point_b[1])

    s = np.abs(0.5*(temp1+temp2+temp3))
    return s


def centrosymmetry_area(polygon, center):
    s = 0
    data_shape = polygon.shape
    n = data_shape[0]

    if n < 3:
        print("<3")
        return 0

    for i in range(0, n-1):
        s = s + triangle_area(polygon[i, :], polygon[i+1, :], center)

    s = s + triangle_area(polygon[n-1, :], polygon[0, :], center)

    return s


def linearity_change(linear1, linear0):
    if linear1 > linear0:
        change = np.log(linear1/linear0)
    else:
        change = np.log(linear0/linear1)*-1

    return np.float(change)


def hist_equalization(data, bins_num=100):
    data_shape = data.shape
    n = data_shape[0]

    data_count = np.zeros((bins_num, 1))
    data_distribution = np.zeros((bins_num, 1))
    data_label = np.zeros((n, 1))

    min_data = min(data)
    max_data = max(data)
    bins_length = (max_data-min_data)/bins_num

    if bins_length == 0:
        return data_label

    for i in range(0, n):
        data_label[i] = int((data[i]-min_data) / bins_length)
        if data_label[i] == bins_num:
            data_label[i] = bins_num - 1

        data_count[int(data_label[i])] = data_count[int(data_label[i])] + 1

    data_distribution[0] = data_count[0]
    for i in range(1, bins_num):
        data_distribution[i] = data_distribution[i-1] + data_count[i]

    label_equalized = np.zeros((bins_num, 1))
    for i in range(0, bins_num):
        label_equalized[i] = (data_distribution[i]-data_count[0])/(n-data_count[0]) * (bins_num-1)

    data_equalized = np.zeros((n, 1))
    for i in range(0, n):
        data_equalized[i] = label_equalized[int(data_label[i])]

    return data_equalized


def linearity(eigen_values):
    (n, d) = eigen_values.shape
    linear = np.zeros((n, 1))

    for i in range(0, n):
        linear[i] = eigen_values[i, 0] / eigen_values[i, 1]

    return linear


def linearity_equalize(linear):
    return hist_equalization(linear)









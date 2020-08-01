import numpy as np
from Tools import Preprocess
from VectorTrans.DRTrans import DRTrans
from VectorTrans.MDSTrans import MDSTrans
from VectorTrans.TSNETrans import TSNETrans
from VectorTrans.PCATrans import PCATrans
from VectorTrans.MDSTransPlus import MDSTransPlus
from VectorTrans.TSNETransPlus import TSNETransPlus
from VectorTrans.CreateJson import JsonFile


def load_data():
    X = np.loadtxt("..\\Data\\data.csv", dtype=np.float, delimiter=",")
    label = np.loadtxt("..\\Data\\label.csv", dtype=np.int, delimiter=",")
    return X, label


def run_example():
    dr_method = 'MDS'  # 'MDS'  't-SNE'  'PCA'  'MDSPlus'  't-SNEPlus'
    X, label = load_data()
    repeat = Preprocess.has_repeat(X)
    if repeat:
        print("Please recheck the input data for duplicate points")
        return
    X = Preprocess.normalize(X)  # Optional
    (n, d) = X.shape

    trans = DRTrans()
    if dr_method == 'MDS':
        trans = MDSTrans(X, label=label, y_init=None, y_precomputed=False)
    elif dr_method == 't-SNE':
        trans = TSNETrans(X, label=label, y_init=None, perplexity=30.0)
    elif dr_method == 'PCA':
        trans = PCATrans(X, label=label)
    elif dr_method == "MDSPlus":
        trans = MDSTransPlus(X, label=label, y_init=None, y_precomputed=False)
    elif dr_method == "t-SNEPlus":
        trans = TSNETransPlus(X, label=label, y_init=None, perplexity=30.0)
    else:
        print("This method is not supported at this time: ", dr_method)
        return

    trans.transform(nbrs_k=20, MAX_EIGEN_COUNT=4, yita=0.1)
    np.savetxt("..\\Data\\"+str(dr_method)+"_Y.csv", trans.Y, fmt='%.18e', delimiter=",")
    if n*d < 1024 ** 3 / 2:
        np.savetxt("..\\Data\\"+str(dr_method)+"_derivative.csv", trans.derivative, fmt='%.18e', delimiter=",")
    json_file = JsonFile(path="..\\Data\\")
    json_file.create_file(trans)


if __name__ == '__main__':
    run_example()




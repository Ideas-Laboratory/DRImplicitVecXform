# DRImplicitVecXform
The source code of paper "**Implicit Multidimensional Projection of Local Subspaces**".

## How to use this code

### Environment

+ This is a Pychram project. If you have installed Pychram in your computer, you can open it in your Pycharm directly. But it's not necessary.
+ This code using [Python 3.7](https://www.python.org/), [numpy](https://numpy.org/) and [scikit-learn](https://scikit-learn.org/). If you want to run this code in your computer, you must install them, and we suggest you use [Anaconda](https://www.anaconda.com/) directly, which has already included them.

### Example

You can find an example in *Main.py*:

```
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
```

+ You could put your data and it's label in the package *Data*. The file *data.csv* contains ``n`` rows and ``D`` columns, and every row is a D-dimensional data point. One thing to note is that there should be **no duplicate** points in your data. *label.csv* is the label of the data.
+ The output files in *Data*:
	+ *XXX_Y.csv* is the result of dimension reduction.
	+ *XXX_derivative.csv* is $\frac{\partial Y}{\partial X}$.
	+ *result.json* and *scale.json* are for visualization.


## Abstract

We propose a visualization method to understand the effect of multidimensional projection on local subspaces, using implicit function differentiation. Here, we understand the local subspace as the multidimensional local neighborhood of data points. Existing methods focus on the projection of multidimensional data points, and the neighborhood information is ignored. Our method is able to analyze the shape and directional information of the local subspace to gain more insights into the global structure of the data through the perception of local structures. Local subspaces are fitted by multidimensional ellipses that are spanned by basis vectors. An accurate and efficient vector transformation method is proposed based on analytical differentiation of multidimensional projections formulated as implicit functions. The results are visualized as glyphs and analyzed using a full set of specifically-designed interactions supported in our efficient web-based visualization tool. The usefulness of our method is demonstrated using various multi- and high-dimensional benchmark datasets. Our implicit differentiation vector transformation is evaluated through numerical comparisons; the overall method is evaluated through exploration examples and use cases.

## 

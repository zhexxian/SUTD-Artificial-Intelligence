from learn import *

# X is an n x d matrix
# Y is an n x 1 matrix
# order indicates order for a polynomial feature space
# method is one of 'gdLin', 'sgdLin', 'gdLog', or 'sgdLog'
# max_iter indicates max number of descent steps
# step_size is passed to gradient descent
# convPlot is a boolean that controls the creation of a plot of function values during descent
# quiet turns off printing

def tclass(X, y, order = 1, method='gdLog', max_iter = 5000, step_size = 0.01,
               convPlot = False, quiet = False):
    phi = polynomialFeaturesN(order)
    phiD = applyFeatureFun(phi, X)

    if method == 'gdLin':
        w, fs, ws = gdLinReg(phiD, y, step_size = step_size / 100, max_iter = max_iter)
    elif method == 'sgdLin':
        w, fs, ws = sgdLinReg(phiD, y, step_size = step_size / 100, max_iter = max_iter)[:3]
    elif method == 'gdLog':
        w, fs, ws = gdLogReg(phiD, y, step_size = step_size, max_iter = max_iter)
    elif method == 'sgdLog':
        w, fs, ws = sgdLogReg(phiD, y, step_size = step_size, max_iter = max_iter)[:3]
    else:
        assert None, 'Unknown method'

    print 'nll', fs[-1], 'num iters', len(fs)
    print w
    eps = .1
    xmin = np.min(X[:,0]) - eps; xmax = np.max(X[:,0]) + eps
    ymin = np.min(X[:,1]) - eps; ymax = np.max(X[:,1]) + eps
    ax = tidyPlot(xmin, xmax, ymin, ymax, xlabel = 'x', ylabel = 'y')
    predictor = makeLogisticRegressor(w, phi)  # sigmoid
    def fizz(xx, yy):
        return predictor(np.matrix([[xx, yy]]))
    res = 30  # resolution of plot
    ima = np.array([[fizz(xi, yi) for xi in np.linspace(xmin, xmax, res)] \
                                for yi in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = 'viridis')  
    plt.colorbar(im)
    colors = [('r' if l == 0 else 'g') for l in y]
    ax.scatter(X[:,0], X[:,1], c = colors, marker = 'o', s=80,
                             edgecolors = 'none')
    if not quiet:
        z = s(phiD*w)
        print y
        print z
    if convPlot:
        pl = len(fs) #min(500,len(fs))
        nax = tidyPlot(0, pl, 0, 3)
        iters = range(0, pl, 100)
        pfs = [float(fs[i]) for i in iters]
        plotData(nax, iters, pfs, style = 'ro-')

# For testing with xor_more data set
def txor_more(order = 1, method = 'gdLog', max_iter = 5000, step_size=0.01,
              convPlot = False):
    X, y = xor_more()
    tclass(X, y, order, method, max_iter, step_size, convPlot)

# Logistic regression using stochastic gradient descent.
# X is a matrix of feature vectors; y is a vector of labels in {0, 1}
# w0 is the initial weight vector
# max_iter is as for gd
# l currently unused, but can be used for regularization
# Returns: final weight vector, a list of scores, list of weight vectors

def sgdLogReg(X, y, l = 0, step_size = 0.01, w0 = None, max_iter = 1000,
                 eps = .00001):
    # w is d by 1; X is n by d
    # return result is d by 1

    # Look at gdLogReg in learn.py
    # define f and df
    # Your code here
    

    # f is the loss
    # df is the gradient/derivative
    if w0 is None: w0 = np.matrix(np.ones(X.shape[1])).T * 0.0000001
    return sgd(X, y, f, df, w0, step_size = step_size, max_iter = max_iter,
               eps = eps)

# Create data set for "Classification via Regresion"
def classReg(addOnes = False):
    X = np.matrix([])                   # your data
    y = np.matrix([]).T                 # your data
    if addOnes:
        X = np.append(np.ones_like(X), X, 1)
    return X, y

# For testing with classReg data set
def tclassReg(order = 1, method = 'gdLin', max_iter = 5000, step_size=0.01,
              convPlot = False):
    X, y = classReg()
    tclass(X, y, order, method, max_iter, step_size, convPlot)

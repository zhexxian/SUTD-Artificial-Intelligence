import pdb
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import operator

######################################################################
#
# Data
#
######################################################################

# Uses data set from Chapter 1 of Bishop, stored in file
# "curvefitting.txt".  They are 10 points drawn from (x, sin(2 pi x))
# with noise added (but I'm not sure how much.)

# If random is not False, it should be an integer, and instead of
# returning data from the file, we will generate a new random data set
# of that size, with 0 mean, 0.2 stdev Gaussian noise.

# if addOnes is true, return: n x 1 matrix X, n x 2 matrix F (with
# column of 1's added) and n x 1 matrix Y.

def getCurveData(addOnes = False, random = False):
    if random:
        X = np.matrix([[i / float(random)] for i in range(random + 1)])
        noise = np.random.normal(scale = 0.2, size = (random+1, 1))
        y = np.matrix([[np.sin(2 * np.pi * X[i,0])] for i in range(X.shape[0])]) + noise
    else:
        data = np.loadtxt('curvefitting.txt')
        X, y = np.matrix(data[0]).T, np.matrix(data[1]).T
    if addOnes:
        F = np.append(np.ones_like(X), X, 1)
        return X, F, y
    else:
        return X, y

def badHat(addOnes = False):
    far = 10
    X = np.matrix([[-1, 0], [-0.1, 0], [0.1, 0], [far, 1], [far, -1]])
    y = np.matrix([[0, 0, 1, 1, 1]]).T
    if addOnes:
        X = np.append(np.ones_like(y), X, 1)
    return X, y

def superSimpleSeparable(addOnes = False):
    X = np.matrix([[2, 3],
                   [3, 2],
                   [9, 10],
                   [10, 9]])
    y = np.matrix([[1, 1, 0, 0]]).T
    if addOnes:
        X = np.append(np.ones_like(y), X, 1)
    return X, y

def superSimpleSeparable2(addOnes = False):
    X = np.matrix([[2, 5],
                   [3, 2],
                   [9, 6],
                   [12, 5]])
    y = np.matrix([[1, 0, 1, 0]]).T
    if addOnes:
        X = np.append(np.ones_like(y), X, 1)
    return X, y

def xor(addOnes = False):
    X = np.matrix([[1, 1],
                   [2, 2],
                   [1, 2],
                   [2, 1]])
    y = np.matrix([[1, 1, 0, 0]]).T
    if addOnes:
        X = np.append(np.ones_like(y), X, 1)
    return X, y

def xor_more(addOnes = False):
    X = np.matrix([[1, 1], [2, 2], [1, 2], [2, 1],
                   [2, 3], [4, 1], [1, 3], [3, 3]])

                   
    y = np.matrix([[1, 1, 0, 0, 1, 1, 0, 0]]).T
    if addOnes:
        X = np.append(np.ones_like(y), X, 1)
    return X, y

def multimodalData(modes = None, numPerMode = 20,
                        numModes = 2,
                        modeCov = np.eye(2, 2)):
    Xs = []
    Ys = []
    if modes is None:
        modes = np.random.multivariate_normal([0, 0], modeCov * 20,
                                                  numModes)
    for (i, mode) in enumerate(modes):
        Xs.extend(np.random.multivariate_normal(mode, modeCov, numPerMode))
        Ys.extend([[i % 2]]*numPerMode)
    return np.matrix(Xs), np.matrix(Ys)
        



######################################################################
#
# Plotting stuff
#
######################################################################

def tidyPlot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plotData(ax, x, y, style = 'ro', c = None, label = None):
    if style is None and c is None:
        ax.plot(x, y, label = label)
    elif style is not None:
        ax.plot(x, y, style, label = label)
    elif c is not None:
        ax.plot(x, y, c = c, label = label)
    plt.show()

# w is (c, a, b)
# ax + by + c = 0
# y = -(a/b) x - (c/b)
def plotLineABC(ax, w, xmin, xmax):
    m = - float(w[1]) / float(w[2])
    b = -float(w[0]) / float(w[2])
    plotFun(ax, lambda x: m*x + b, xmin, xmax)

# w is a (1 x 2) matrix
def plotLine(ax, w, xmin, xmax, nPts = 100):
    b = float(w[0])
    m = float(w[1])
    plotFun(ax, lambda x: m*x + b, xmin, xmax, nPts)

def plotFun(ax, f, xmin, xmax, nPts = 100, label = None):
    x = np.linspace(xmin, xmax, nPts)
    y = np.vstack([f(np.matrix([[xi]])) for xi in x])
    ax.plot(x, y, label = label)
    plt.show()


def smooth(n, vals):
    # Run a box filter of size n
    x = sum(vals[0:n])
    result = [x]
    for i in range(n, len(vals)):
        x = x - vals[i-n] + vals[i]
        result.append(x)
    return result

######################################################################
#
# Test NN
#
######################################################################

import modules
reload(modules)
import data_io

def predictionErrors(nn, X, y, verbose=False):
    errors = 0
    n = X.shape[0]
    for i in range(n):
        pred = nn.forward(np.asarray(X[i,:]))
        ypred = np.argmax(pred)
        if not ypred==y[i]:
            errors += 1
            if verbose:
                print i, 'prediction', pred, ypred, 'actual', int(y[i])
    accuracy = 1.0 - (float(errors)/n)
    print 'Prediction accuracy', accuracy
    return errors



def listify(a):
    l = []
    if len(a.shape) == 1:
        return a.tolist()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            l.append(a[i,j])
    return l

def testNN(iters=10000, width=3, learning_rates = [0.005], std=True):
    # Test cases
    # X, y = xor()
    # X, y = xor_more()
    X, y = multimodalData(numModes = 4)
    # X, y = sklearn.datasets.make_moons(noise=0.25)
    # X, y = sklearn.datasets.make_classification()
    # y has 2 classes (0, 1)

    # Map into 2 softmax outputs
    nclass = len(set(listify(y)))
    def unary(yi): return [(1 if i == yi else 0) for i in range(nclass)]
    Y = np.array([unary(yi) for yi in y])
    #build a network
    nd = X.shape[1]                     # number of features
    u = width
    nn = modules.Sequential([modules.Linear(nd,u),
                             modules.Tanh(),
                             modules.Linear(u,u),
                             modules.Tanh(),
                             modules.Linear(u,nclass),
                             modules.SoftMax()])

    # train the network.
    errors = []
    for lrate in learning_rates:
        nn.clean()
        # Default does not do learning rate decay or early stopping.
        #print ('X,Y,',X,Y)

        nn.train2(np.asarray(X),np.asarray(Y), batchsize = 1, iters = iters,
                  lrate=lrate,
                  lrate_decay_step = 100000
                  )
        errors.append((lrate, predictionErrors(nn, X, y)))
    print 'Errors for learning rates', errors

    # Plot the last result
    if nd > 2: return                   # only plot two-feature cases
    eps = .1
    xmin = np.min(X[:,0]) - eps; xmax = np.max(X[:,0]) + eps
    ymin = np.min(X[:,1]) - eps; ymax = np.max(X[:,1]) + eps
    ax = tidyPlot(xmin, xmax, ymin, ymax, xlabel = 'x', ylabel = 'y')
    def fizz(x1, x2):
        y =  nn.forward(np.array([x1, x2]))
        return y[0,1]                   # class 1
    
    res = 30  # resolution of plot
    ima = np.array([[fizz(xi, yi) for xi in np.linspace(xmin, xmax, res)] \
                                for yi in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = 'viridis' if std else 'jet')  
    plt.colorbar(im)
    if std:
        colors = [('r' if l == 0 else 'g') for l in y]
        ax.scatter(X[:,0], X[:,1], c = colors, marker = 'o', s=80,
                   edgecolors = 'none')
    else:
        pinds=np.where(y==0)
        ninds=np.where(y==1)
        plt.plot( X[pinds[0],0],X[pinds[0],1], 'ob')    
        plt.plot( X[ninds[0],0],X[ninds[0],1], 'or')    

def testNN2(runs=1, width=3, data="iris.txt",
            iters=20000, std=True, trainPct = 0.666):
    if data == 'gauss':
        X, y = multimodalData(numModes = 4, numPerMode=30)
        # X, y = sklearn.datasets.make_classification()
        XY = np.asarray(np.hstack([X,y.reshape((X.shape[0],1))]))
    else:
        XY = data_io.read(data)
    nclass = len(set(XY[:, -1]))    # number of classes
    # y has nclass classes (0, ..., nclass-1)
    def unary(yi): return [(1 if i == yi else 0) for i in range(nclass)]
    # build a network
    u = width
    nn = modules.Sequential([modules.Linear(XY.shape[1]-1,u),
                             modules.Tanh(),
                             modules.Linear(u,u),
                             modules.Tanh(),
                             modules.Linear(u,nclass),
                             modules.SoftMax()])
    results = {False: [], True: []}
    for run in range(runs):
        Xtrain, ytrain, Xtest, ytest = splitByClass(XY, trainPct)
        # Map into n softmax outputs
        Ytrain = np.array([unary(yi) for yi in ytrain])
        for rms in (False, True):
            # train the network.
            nn.clean()
            nn.train2(np.asarray(Xtrain),np.asarray(Ytrain),
                      batchsize = 1, iters = iters,
                      lrate_decay_step = 1000,
                      rms=rms,
                      momentum=(0.9 if rms else None))
            errors = predictionErrors(nn, Xtest, ytest)
            accuracy = 1.0 - (float(errors)/Xtest.shape[0])
            print 'RMS', rms, 'Prediction accuracy', accuracy
            results[rms].append(accuracy)
    print 'Results', results
    print 'Average accuracy', 'rms=False', sum(results[False])/runs, 'rms=True', sum(results[True])/runs, 

# Split into train and test sets keeping the same class distribution
def splitByClass(XY, trainPct):
    nclass = len(set(XY[:, -1]))    # number of classes
    byClass = [XY[XY[:, -1] == c] for c in range(nclass)]
    train = []
    test = []
    for XYc in byClass:
        n = XYc.shape[0]
        ntr = int(trainPct*n)
        np.random.shuffle(XYc)
        train.append(XYc[:ntr,:])
        test.append(XYc[ntr:,:])
    return np.vstack([X[:, :-1] for X in train]), np.vstack([X[:, -1:] for X in train]), \
           np.vstack([X[:, :-1] for X in test]), np.vstack([X[:, -1:] for X in test]),

print 'Loaded learn.py'

if __name__=='__main__':
    testNN(iters=10000, width=3, learning_rates = [0.005], std=True)
    plt.show(block=True)

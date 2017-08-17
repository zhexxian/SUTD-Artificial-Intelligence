import modules_mnistsimple
import utils_mnist
import numpy

import matplotlib.pyplot as plt

ADAM = False

def tester():

  mnistpath='./mnist'


  nn = modules_mnistsimple.Network([
    modules_mnistsimple.Linear(784,300, 'mlp2/l1'),modules_mnistsimple.ReLU(),
    modules_mnistsimple.Linear(300,100, 'mlp2/l2'),modules_mnistsimple.ReLU(),
    modules_mnistsimple.Linear(100,10, 'mlp2/l3'),
  ])

  Xtr,Ytr=utils_mnist.getMNISTtrain(seed=None,path=mnistpath)

  numall=10000
  numval=3000
  Xte,Yte=utils_mnist.getMNISTsample(N=numall,seed=None,path=mnistpath)
  numtest=numall-numval


  print Xtr.shape,Ytr.shape,Xte.shape,Yte.shape

  if ADAM:
    nn.train_softmaxontop( X=Xtr, Y=Ytr, Xval=Xte[1:numval,:], Yval=Yte[1:numval,:], batchsize=25, iters=10000, lrate=0.001, useAdam=True, status=250, transform=None)
  else:
    nn.train_softmaxontop( X=Xtr, Y=Ytr, Xval=Xte[1:numval,:], Yval=Yte[1:numval,:], batchsize=25, iters=10000, lrate=0.0005, lrate_decay='phase',lrdecay_step = 3000, lrdecay_factor = 0.5 , status=250, transform=None)

  
  Xtimg=Xte[numval:,:]
  lbs=Yte[numval:,:]
  Z=nn.forward(Xtimg)
  print Z.shape

  corr=0
  
  predictions=[ [] for i in range(10) ]
  imageindices=[[] for i in range(10) ]
  
  for i in xrange(numtest):
    plb=numpy.argmax(Z[i,:])
    print plb

    predictions[int(plb)].append(Z[i,plb])
    imageindices[int(plb)].append(numval+i)

    corr+= (plb == numpy.argmax(lbs[i,:])) *1.0/numtest
    
  print 'Accuracy on TEST SET',corr*100


  classnames=['Zeros','Ones','Twos','Threes','Fours','Fives','Sixes','Sevens','Eights','Nines']

  for c in range(10):
    order=numpy.argsort(-numpy.array(predictions[c]))    
    orderarray=numpy.zeros((order.size))
    for i in range(order.size):
        orderarray[i]= imageindices[c][order[i]]
     
    rankindstart=0

    #choose here another value if you want to see digits with lower ranks
    
    titlestring= 'Top-ranked '+ classnames[c]+' starting at rank '+str(rankindstart)

    print 'starting to visualize top-ranked digits for class '+str(c)

    utils_mnist.showsomerankedmnist(rankindstart,orderarray,Xte,titlestring)


  plt.show()

if __name__=='__main__':
  tester()




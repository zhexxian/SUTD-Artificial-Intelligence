import numpy as np
import matplotlib.pyplot as mpl

def datagen(n):
  c=np.random.randint(1,3,size=(n))
  us=np.random.normal(loc=0,scale=1,size=(2,n))
  
  A=np.zeros((2,2))
  A[0,0]=np.cos(np.pi/4.)
  A[1,1]=A[0,0]
  A[0,1]=-np.sin(np.pi/4)
  A[1,0]=-A[0,1]
  B=np.zeros((2,2))
  B[0,0]=3
  B[1,1]=1
  A=np.dot(A,B)
  
  xs=np.dot(A,us)
  ys=np.random.rand(n)
  
  mu2=np.zeros((2,1))
  mu2[0]=2.5
  
  print(xs.shape,mu2.shape,xs[:,0].shape)
  for i in range(n):
    if(c[i]==2):
      xs[:,i]+=mu2.reshape((2))
      ys[i]=  (ys[i]>0.2)
    else:
      ys[i]=  (ys[i]>0.8)
  
  
  return xs,ys,c
  
def plot1(n):
  xs,ys,c=datagen(n)
  
  inds=np.where(ys==0)
  print(inds[0].size)
  mpl.plot(xs[0,inds[0]],xs[1,inds[0]],'r.')
  inds=np.where(ys==1)
  mpl.plot(xs[0,inds[0]],xs[1,inds[0]],'b.')  
  
  mpl.show(block=True)
  
def knn(traindata, trainlabels,x):
  
  dists=np.ones(traindata.shape[1])
  minind=0
  mindist=np.dot(traindata[:,minind]-x,traindata[:,minind]-x)
  for i in range(1,traindata.shape[1]):
    dists[i]= np.dot(traindata[:,i]-x,traindata[:,i]-x)
    if(  dists[i] < mindist):
      minind=i
      mindist=dists[i]
      
  return trainlabels[minind]

def cls(n):
  xs,ys,c=datagen(n)
  
  xt,yt,ct=datagen(n)
  
  
  err1=0
  for i in range(xt.shape[1]):
     pred=knn(xs, ys,xt[:,i])
     err1+=  (pred!=yt[i]) /float(xt.shape[1])
  print ('err1',err1)
  
  err2=0
  for i in range(xt.shape[1]):
     pred= np.dot(-np.array([-1,1]),xt[:,i])-1.25 
     pred=pred>0
     err2+=  (pred!=yt[i]) /float(xt.shape[1])
  print ('err2',err2)  
  
if __name__=='__main__':
  #plot1(1000)
  cls(1000)


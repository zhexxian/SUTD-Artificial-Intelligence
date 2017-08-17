'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.1
@copyright: Copyright (c)  2015, Sebastian Bach, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller
@license : BSD-2-Clause
'''

import numpy as np
na = np.newaxis
import copy
import sys

RMSProp = True
mu = None				# momentum

# -------------------------------
# Modules for the neural network
# -------------------------------
class Module:
	'''
	Superclass for all computation layer implementations
	'''
	
	def __init__(self): pass
	def update(self, lrate): pass
	def clean(self): pass
	def backward(self,DY): return DY
	def train(self, X, Y, batchsize, iters, lrate, status, shuffle_data): pass
	def forward(self,X): return X

# Geoff Hinton says: "If a hidden unit has a big fan-in, small changes
# on many of its weights can cause the learning to overshoot.  We
# generally want smaller incoming weights when the fan-in is big, so
# initialize the weights to be proportional to sqrt(fan-in).  We can
# also scale the learning rate the same way."

# -------------------------------
# Linear layer
# -------------------------------
class Linear(Module):
	# do this after finining the previous one, tanh

	# need to sum out the batch size, even though the batch size is set to 1
	'''
	Linear Layer
	'''

	def __init__(self,m,n):
		'''
		Initiates an instance of a linear computation layer.
		
		Parameters
		----------
		m : int
			input dimensionality
		n : int
			output dimensionality
			
		Returns
		-------
		the newly created object instance
		'''
		
		self.m = m
		self.n = n
		self.B = np.zeros([self.n])
		# See Hinton quote above
		self.W = np.random.normal(0,1.0*m**(-.5),[self.m,self.n])
		self.RW = None		# for RMSProp
		self.RB = None		# for RMSProp
		self.vW = None		# velocity
		self.vB = None		# velocity

	def forward(self,X):
		self.X = X #this value will persists, as it is class variable
		self.Y = np.dot(X,self.W)+self.B


		return self.Y

	def backward(self,DY):
		pass

		#your code here: implement backward

                # Y.shape=DY.shape=(batchsize,outputlayersize)
                # X.shape= (batchsize,inputlayersize)
                # W.shape= (inputlayersize,outputlayersize)
                # B.shape= (outputlayersize,) (1-dim python array)

                #self.DW must have same shape as self.W, take DY and multiply it with the proper derivative - it must sum over all elements in the batchsize
                #self.DB must have same shape as self.B, take DY and multiply it with the proper derivative - it must sum over all elements in the batchsize
                #backward(self,DY) should return the DY for the next layer, that is "DY from the input weighted with DY/DX, multiply the output with *self.m**.5/self.n**.5
                	# newDY = DY * output with respect to the input

	def update(self, lrate):
		self.W -= lrate*self.dW/self.m**.5
		self.B -= lrate*self.dB/self.m**.25	


	def clean(self):
		self.X = None
		self.Y = None
		self.dW = None
		self.dB = None
		self.RW = None
		self.RB = None
		self.vW = None
		self.vB = None


# -------------------------------
# Sigmoid layer
# -------------------------------
class Sigmoid(Module):
	'''
	Sigmoid Layer
	'''

	def forward(self,X):
		self.Y = 1.0 / (1.0 + np.exp(-X))
		return self.Y
	

	def backward(self,DY):
		return DY*(self.Y * (1 - self.Y))


	def clean(self):
		self.Y = None

# -------------------------------
# Tanh layer
# -------------------------------
class Tanh(Module):
		# good code reference: http://neuralnetworksanddeeplearning.com/chap2.html
	'''
	Tanh Layer
	'''

	def forward(self,X):
		pass
		#implement tanh neuron here

                # Y.shape=DY.shape=(batchsize,outputlayersize)
                # X.shape= (batchsize,inputlayersize)
                # for an activation layer: inputlayersize=outputlayersize
	
			#just comput tanh of input x; what to store as self.something? for use in backward pass

	def backward(self,DY):
		pass
		#implement the backpropagation rule for a tanh neuron here

                # Y.shape=DY.shape=(batchsize,outputlayersize)
                # X.shape= (batchsize,inputlayersize)
                # for an activation layer: inputlayersize=outputlayersize

		#implement the backpropagation rule for a tanh neuron here
                #backward(self,DY) should return the DY for the next layer, that is "DY from the input weighted with DY/DX, here must do it for every element in the batch separately,
                # no summing over batchsize because Y.shape=DY.shape=(batchsize,outputlayersize), X.shape= (batchsize,inputlayersize) - batch size is kept here

	def clean(self):
		self.Y = None


# -------------------------------
# Softmax layer
# -------------------------------
class SoftMax(Module):
	'''
	Softmax Layer
	'''
	
	def forward(self,X):
		
		if( len(X.shape)==1):
			self.X = X.reshape((1,X.shape[0]))
		else:
			self.X = X
		
		maxvals=np.max(self.X,1)
		maxarray=np.tile(maxvals.reshape((self.X.shape[0],1)), (1,self.X.shape[1]))

		self.Y = np.exp(self.X-maxarray) / np.exp(self.X-maxarray).sum(axis=1)[:,na]
		return self.Y


	def clean(self):
		self.X = None
		self.Y = None


# -------------------------------
# Sequential layer
# -------------------------------   
class Sequential(Module):
	'''
	Top level access point and incorporation of the neural network implementation.
	Sequential manages a sequence of computational neural network modules and passes
	along in- and outputs.
	'''

	def __init__(self,modules):
		'''
		Constructor
				
		Parameters
		----------
		modules : list, tuple, etc. enumerable.
			an enumerable collection of instances of class Module
		'''
		self.modules = modules

	def forward(self,X):
		'''
		Realizes the forward pass of an input through the net
				
		Parameters
		----------
		X : numpy.ndarray
			a network input.
		
		Returns
		-------
		X : numpy.ndarray
			the output of the network's final layer
		'''
		
		for m in self.modules:
			X = m.forward(X)
		return X


	def train(self, X, Y,  Xval = [], Yval = [],  batchsize = 25, iters = 10000, lrate = 0.005,
		  status = 250, shuffle_data = True, rms = False, momentum=None):
		''' 		
			X the training data
			Y the training labels
			
			Xval some validation data
			Yval the validation data labels
			
			batchsize the batch size to use for training
			iters max number of training iterations . TODO: introduce convergence criterion
			lrate the learning rate
			status number of iterations of silent training until status print and evaluation on validation data.
			shuffle_data permute data order prior to training
		'''
		global RMSProp, mu
		RMSProp = rms
		self.mu = momentum

		if Xval == [] or Yval ==[]:
			Xval = X
			Yval = Y
		
		N,D = X.shape
		if shuffle_data:
			r = np.random.permutation(N)
			X = X[r,:]
			Y = Y[r,:]
			
		for i in xrange(iters):
			samples = np.mod(np.arange(i,i+batchsize),N)
			Ypred = self.forward(X[samples,:])
			self.backward(Ypred - Y[samples,:])
			self.update(lrate)
			
			if i % status == 0:
				Ypred = self.forward(Xval)
				acc = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Yval, axis=1))
				print 'Accuracy after {0} iterations: {1}%'.format(i,acc*100) 
		
		
		
		
		
		
	def backward(self,DY):
		for m in self.modules[::-1]:
			DY = m.backward(DY)
		return DY
	
	def update(self,lrate):
		for m in self.modules: m.update(lrate)
	
	def clean(self):
		'''
		Removes temporary variables from all network layers.
		'''
		for m in self.modules: m.clean()


	#new add from sebastian
	def train2(self, X, Y,  Xval = [], Yval = [],  batchsize = 25, iters = 10000,
		   lrate = 0.005,  lrate_decay_step= 1000, lrate_decay_factor = 0.9, status = 250,
		   convergence = -1, transform = None, rms = False, momentum = None):
		'''        
		    X the training data
		    Y the training labels
		   
		    Xval some validation data
		    Yval the validation data labels
		   
		    batchsize the batch size to use for training
		    iters max number of training iterations.
		    lrate the learning rate is adjusted during training with increased model performance. See lrate_decay
		    lrate_decay_step controls how often rate decay is applied
		    lrate_decay_factor controls decay
		    status number of iterations of silent training until status print and evaluation on validation data.
		    convergence number of consecutive allowed status evaulations with no more model improvements until we accept the model has converged. Set <=0 to disable.
		    transform a function taking as an input a batch of training data sized [N,D] and returning a batch sized [N,D] with added noise or other various data transformations.
		        default value is None for no transformation.
		'''
       
		untilConvergence = convergence
		bestAccuracy = 0.0
		bestLayers = copy.deepcopy(self.modules)
		global RMSProp, mu
		RMSProp = rms
		mu = momentum
		       
		print 'Convergence', convergence

		N,D = X.shape      

                print('N,D',N,D)
 
		for i in xrange(iters):
		   
		    #the actual training: pick samples at random
		    r = np.random.permutation(N)
		    samples = r[0:batchsize]
		   
		    if transform == None:
		        batch = X[samples,:]
		    else:
		        batch = transform(X[samples,:])
		   
		    #forward and backward propagation steps with parameter update

                    #print('batch.shape ', batch.shape)
		    Ypred = self.forward(batch)

		    self.backward(Ypred - Y[samples,:])
		    self.update(lrate)
		   
		    #periodically evaluate network and optionally adjust learning rate or check for convergence.
		    if (i+1) % status == 0:
		        Ypred = self.forward(X)
		        acc = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Y, axis=1))
		        print 'Accuracy after {0} iterations: {1}%'.format(i+1,acc*100)
		        if not Xval == [] and not Yval == []:
		            Ypred = self.forward(Xval)
		            acc_val = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Yval, axis=1))
		            print 'Accuracy on validation set: {1}%'.format(i+1,acc_val*100)
		       
		        if acc > bestAccuracy:
		            print '    New optimal parameter set encountered. saving....'
		            bestAccuracy = acc
		            bestLayers = copy.deepcopy(self.modules)
		            untilConvergence = convergence
			else:
		            untilConvergence-=1
		            if untilConvergence == 0 and convergence > 0:
		                print '    No more recorded model improvements for {0} evaluations. Accepting model convergence.'.format(convergence)
		                break

		    if (i+1) % lrate_decay_step == 0:
			    print 'lrate', lrate
			    lrate *= lrate_decay_factor
	       
		#after training, either due to convergence or iteration limit
		print 'Setting network parameters to best encountered network state with {0}% accuracy.'.format(bestAccuracy*100)
		self.modules = bestLayers





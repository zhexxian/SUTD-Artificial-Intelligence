
import numpy
import matplotlib.pyplot as plt


class quadform2d:

        def __init__(self):
                pass
                #flat valley along dim0, steep along dim1
                self.H=numpy.array([[0.1, 0.0],[0.0, 10.0]])

                print 'self.H.shape', self.H.shape


                self.momentum=0.9


                self.oldDX=None
                self.Xhist=numpy.array((2,1))


                self.oldnorms=None
                self.rmspropconst=0.9
                self.stab=1e-6


        def forward(self,X):
                # X should have shape (2,1)
                # min is at (0,0)
                self.X=X
                val= 0.5*numpy.dot(numpy.dot( self.X.T,self.H ), self.X) # x.T \cdot H \cdot X
                return val

        def backward(self, DY):
                self.DX= numpy.dot(self.H , self.X)
                #needs not to return anything we do not do backprop


        def update_s(self, lr):


                self.X -= lr * self.DX

        ###############################################################################################
        # Students please complete this one
        ##############################################################################################
        def update_a(self, lr):

                if self.oldnorms is None:
                        self.oldnorms= 0 
                        self.updnorm = 0
                        updX =  self.DX * 0.01
                else:
                        #compute RMSprop part of adadelta
         
                        self.X -= lr * updX
                        #recompute self.updnorm


        def train_s(self, Xinit, iters = 10000,lrate = 0.005, stopcrit=1e-4, lratedecrease= 'None', errormode='cheat'):

                itercount=0
                converged=False
                oldval=-1

                self.X=Xinit

                while (False == converged):
        

                        Ypred = self.forward(self.X)

                        #print self.X.T,Ypred

                        self.backward(0)
                        if lratedecrease == 'None':
                                self.update_s(lrate)
                        else:
                                fac= int(itercount/200)
                                mul= max(0.1**fac,1e-5)

                                self.update_s(lrate*mul)

                        if(itercount==0):
                                self.Xhist=self.X
                        else:
                                self.Xhist=numpy.hstack( (self.Xhist,self.X))

                        if oldval >= 0:
                                if errormode == 'cheat':
                                        diff= abs(Ypred)
                                else:
                                        diff= abs(Ypred-oldval)
                                #print itercount,'change in objective value: '+str(diff)+' abs value: '+str(Ypred)
                                if( diff < stopcrit):
                                        converged=True
                                        print 'CONVERGED in '+str(itercount)+ ' steps'  , diff, Ypred
                        else:
                                diff=1000000

                        oldval=Ypred
                        itercount+=1
                     
                        if( (itercount > iters) and (False==converged)):
                                converged=True
                                print 'terminated in '+str(itercount)+ ' steps, max number of iterations,  reached', Ypred      
        

        def train_m(self, Xinit, iters = 10000,lrate = 0.005, stopcrit=1e-4, lratedecrease= 'None', errormode='cheat'):

                itercount=0
                converged=False
                oldval=-1

                self.X=Xinit

                while (False == converged):
        
                        
                        Ypred = self.forward(self.X)
                


                        self.backward(0)

                        if lratedecrease == 'None':
                                self.update_a(lrate)
                        else:
                                fac= int(itercount/200)
                                mul=max(0.1**fac,1e-5)
                                
                                self.update_a(lrate* mul)

                        if(itercount==0):
                                self.Xhist=self.X
                        else:
                                self.Xhist=numpy.hstack( (self.Xhist,self.X))   


                        if oldval >= 0:
                                if errormode == 'cheat':
                                        diff= abs(Ypred)
                                else:
                                        diff= abs(Ypred-oldval)

                                #print itercount,' change in objective value: '+str(diff)+' abs value: '+str(Ypred)
                                if( diff < stopcrit):
                                        converged=True
                                        print 'CONVERGED in '+str(itercount)+ ' steps', diff, Ypred     
                        else:
                                diff=1000000

                        oldval=Ypred
                        itercount+=1
                     
                        if( (itercount > iters) and (False==converged)):
                                converged=True
                                print 'terminated in '+str(itercount)+ ' steps, max number of iterations reached', Ypred        



def tester5():
        

        zval=0.1

        Xinit=numpy.array([100.0,zval]).reshape((2,1))
        
        #lrate2=0.1
        #lrate2=0.01
        lrate2=0.19
        
        qf=quadform2d()
        start=Xinit
        qf.train_s(start,iters = 10000,lrate = lrate2, stopcrit=1e-6, lratedecrease= 'None', errormode='cheat')

        plt.figure(1)
        plt.plot( qf.Xhist[0,:] , qf.Xhist[1,:],'.')

        
        Xinit=numpy.array([1.0,zval]).reshape((2,1))
        start=Xinit
        qf.oldDX=None   

        lrate2=1
        qf.train_m(start,iters = 10000,lrate = lrate2, stopcrit=1e-6, lratedecrease= 'None', errormode='cheat')

        plt.figure(2)
        plt.plot( qf.Xhist[0,:] , qf.Xhist[1,:],'.')



        plt.show()


def tester6():
        

        zval=0.1

        Xinit=numpy.array([100.0,zval]).reshape((2,1))
        
        lrate2=0.19
        
        qf=quadform2d()
        start=Xinit
        qf.train_s(start,iters = 10000,lrate = lrate2, stopcrit=1e-6, lratedecrease= 'None', errormode='cheat')

        plt.figure(1)
        plt.plot( qf.Xhist[0,:] , qf.Xhist[1,:],'.')

        
        Xinit=numpy.array([1.0,zval]).reshape((2,1))
        start=Xinit
        qf.oldDX=None   

        lrate2=1
        qf.train_m(start,iters = 10000,lrate = lrate2, stopcrit=1e-6, lratedecrease= 'phase200', errormode='cheat')

        plt.figure(2)
        plt.plot( qf.Xhist[0,:] , qf.Xhist[1,:],'.')



        plt.show()


def tester7():
        

        zval=0.1

        Xinit=numpy.array([100.0,zval]).reshape((2,1))
        
        lrate2=0.19
        
        qf=quadform2d()
        start=Xinit
        qf.train_s(start,iters = 10000,lrate = lrate2, stopcrit=1e-6, lratedecrease= 'None', errormode='cheat')

        plt.figure(1)
        plt.plot( qf.Xhist[0,:] , qf.Xhist[1,:],'.')

        
        Xinit=numpy.array([1.0,zval]).reshape((2,1))
        start=Xinit
        qf.oldDX=None   

        lrate2=10
        qf.train_m(start,iters = 10000,lrate = lrate2, stopcrit=1e-6, lratedecrease= 'phase200', errormode='cheat')

        plt.figure(2)
        plt.plot( qf.Xhist[0,:] , qf.Xhist[1,:],'.')



        plt.show()


if __name__=='__main__':
        tester5() # standard and Adadelta with same learning rate, learning rate constant
        #tester6() # standard and Adadelta with same learning rate, learning rate phased
        #tester7() # standard and Adadelta with different learning rate, learning rate phased



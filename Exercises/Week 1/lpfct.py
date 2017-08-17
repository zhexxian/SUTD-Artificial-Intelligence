
# a note on python functions

# we want to return a function function(x) that takes an input x, and outputs whatever
# this function depends on Parameters params

def codethatgenerates_a_pythonfunction(params):

  def function(x):
    #can use params here, they are visible here!
    somevalue= x * params[0,0]
    return somevalue
  # definition of function ends here
    
  return function
# definition of  codethatgenerates_a_pythonfunction ends here 
# -- it returns a function that is callable as function(x) but depends on params






#a note on  np.matrix

#initialize a matrix of some shape (a,b):
a=np.matrix(   np.zeros((a,b))   )


#np.array multiplication:
np.dot(arr1,arr2)

#np.matrix multiplication:
arr1 * arr2



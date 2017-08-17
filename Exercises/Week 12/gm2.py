import pdb
import operator
def mul(seq):
    return reduce(operator.mul, seq, 1)

print 'This version of gm for use during first lecture.'

class Potential:
    # potential is a class which can store joint or conditional probabilities
    #
    # variables: list of strings naming the variables
    # pot: dictionary mapping tuples of variable values to potential value
    def __init__(self, variables, pot):
        self.vars = variables
        self.indices = dict(zip(variables, range(len(variables)))) # dictionary = dict(zip(keys, values))
        # maps variable names to position indices in pot, e.g. A->0, B->1
        self.pot = pot # if have joint over A,B, with values {0,1} and {2,3} then pot[(0,2)] must have the value of P(A=0,B=2) -- self.indices store the position indices for variables A,B,C. so that one can know that n pot[(0,2)] the first element corresponds to variable A, and must mean therefore A=0, the second element orresponds to variable B, and must mean therefore B=2
        # so pot is a dictionary that maps tuples of variable values

    def __str__(self):
        return 'Potential('+str(self.vars)+','+str(self.pot)+')'

    # vt is a tuple of values; return the associated potential value
    # return 0 if vt is not explicitly represented in self.pot
    def valTuple(self, vt): #vt can be (0,2)
        return self.pot[vt] if vt in self.pot else 0.0

    # Return a list of all elements (tuples such as  (0,2),(0,3),(1,2),(1,3) ) that have weight > 0 in this potential
    def support(self):
        return [k for (k, v) in self.pot.items() if v > 0]

    # assign is a dictionary mapping variable names to values, e.g. assign['A']=1,assign['B']=3
    # the code check self.vars for 'A', 'B' and fetches the values for them into the list [assign[var] for var in self.vars]
    # so if assign['A']=1,assign['B']=3, then it returns via valTuple self.pot[(1,3)] meaning Pot(A=1,B=3)
    #return the associated potential valu.
    def val(self, assign):
        return self.valTuple(tuple([assign[var] for var in self.vars]))

    # Product of two instances of Potential is a new Potential defined
    # on the union of the variables of self and other
    def mul(self, other):
        # Three sets of vars: only in self, in both, only in other
        selfOnly = set(self.vars).difference(set(other.vars))
        otherOnly = list(set(other.vars).difference(set(self.vars)))
        both = set(self.vars).intersection(set(other.vars))
        # keep whole tuple from self; add some indices from other
        otherIndices = [other.indices[v] for v in otherOnly]
        newPot = {} #empty dict
        for e1 in self.support(): # loop over all tuples with pos value in potential from self
            for e2 in other.support(): #loop overall tuples with pos value in potential from other
                if self.agrees(other, e1, e2, both): # e1 is a tuple with pos value in potential from self
                    # use only tuples for multiplication where for variables in both the values are same in e1 and in e2. can if for self A,B encodes tuples (0,1) and B,C encodes (x,y), then multiply only if
                    # x=1 because in first tuple B=1, so in second tuple B=x must have B=1 too 
                    newElt = tuple(list(e1) + [e2[i] for i in otherIndices])
                    newPot[newElt] = self.valTuple(e1) * other.valTuple(e2)
        return Potential(self.vars + otherOnly, newPot)


    # vs is a list of variable names
    # Assume: tuple1 is an assignment of the variables in self, tuple
    # 2 is an assignment of variables in other.  Return True if they
    # agree on the values of the variables in vs
    def agrees(self, other, tuple1, tuple2, vs):
        for v in vs:
            if tuple1[self.indices[v]] != tuple2[other.indices[v]]:
                return False
        return True

    # cVars is a list of variable names
    # cVals is a list of the same length of values for those variables
    # Treat self as a joint probability distribution, and this as the
    # operation of conditioning on the event cVars = cVals
    # - select out entries for which cVars = cVals
    # - remove cVars from the potential
    # - sum potential values if there are duplicate entries
    # - renormalize to obtain a distribution
    # Returns a new instance of Potential defined on previous vars minus cVars
    def condition(self, cVars, cVals):
        newPot = {}
        indices = [self.indices[v] for v in cVars]
        for e in self.support():
            if all(e[i] == v for (i, v) in zip(indices, cVals)):
                newPot[removeIndices(e, indices)] = self.pot[e]
        return Potential(removeIndices(self.vars, indices), newPot).normalize()

    # qVars is a list of variable names
    # Sum out all other variables, returning a new potential on qVars
    def marginalize(self, qVars):
        newPot = {}
        indices = removeVals(range(len(self.vars)),
                             [self.indices[v] for v in qVars])
        for e in self.support():
            newE = removeIndices(e, indices)
            addToEntry(newPot, newE, self.valTuple(e))
        return Potential(qVars, newPot)

    # Divide through by sum of values; returns a new Potential on the
    # same variables with potential values that sum to 1 over the
    # whole domain.
    def normalize(self):
        total = sum(self.pot.values())
        newPot = dict([(v, p/total) for (v, p) in self.pot.items()])
        return Potential(self.vars, newPot)

# Convenient abbreviation
P = Potential

# Useful as the multiplicitive identity:  p.mul(iPot) = p 
iPot = Potential([], {tuple() : 1.0})

######################################################################
# Bayesian networks
######################################################################

class BNNode:
    # name is a string naming the variable
    # parents is a list of strings naming parent variables
    # cpd is an instance of Potential, defined on variables [name] + parents
    # It needs to be a well-formed conditional probability
    # distribution, so that for each value v of name,
    # sum_{values of v} cpd([v] + values of parents) = 1
    def __init__(self, name, parents, cpd):
        self.name = name
        self.parents = parents
        self.cpd = cpd

class BN:
    # takes a list of nodes
    def __init__(self, nodes):
        self.vars = [n.name for n in nodes]
        # LPK: Check to be sure all parents are in network
        self.nodes = nodes

    # assign is a dictionary from variable names to values, with an
    # entry for every variable in the network
    # Returns probability of that assignment
    def prob(self, assign):
        pass

    # Create a joint probability distribution
    # Returns a potential reprsenting the joint distribution, defined
    # over all the variables in the network
    def joint(self):
        j = reduce(Potential.mul, [n.cpd for n in self.nodes], iPot)
        assert 1-1e-8 < sum(j.pot.values()) < 1 + 1e-8
        return j

    # queryVars is a list of variable names
    # eVars is a list of variable names
    # eValues is a list of values, one for each of eVars
    # Returns a joint distribution on the query variables representing
    # P(queryVars | eVars = eValues)
    def query(self, queryVars, eVars = [], eValues = []):
        # your code here
        pass

######################################################################


# xs is a tuple (or list) of items and indices is a list of indices
# returns a new tuple containing only those items whose indices are
#  not in the list
def removeIndices(xs, indices):
    return tuple([xs[i] for i in range(len(xs)) if not i in indices])

# xs is a tuple (or list) of items and vals is a list of values
# returns a new tuple containing only those items whose indices are
# not in the list.  Use this instead of set difference because we want
# maintain the order of the remaining xs
def removeVals(xs, vals):
    return tuple([x for x in xs if x not in vals])

# Assuming d is a dictionary mapping elements to numeric values
# Adds e to the dictionary if it is not already there
# Increments the value of e by v
def addToEntry(d, e, v):
    if not e in d: d[e] = 0
    d[e] += v

######################################################################
# Test cases
######################################################################
    
# Wet grass # R- rain, S-sprinkler, J - jings lawn wet, T-Tongs lawn wet. sprinkler affects only tongs lawn, rain affects both
wg = BN([BNNode('R', [], P(['R'], {(0,) : .8, (1,) : .2})),
         BNNode('S', [], P(['S'], {(0,) : .9, (1,) : .1})),
         BNNode('J', ['R'], 
               P(['J', 'R'], 
                  {(0, 0) : 0.8, (1, 0) : 0.2, (0, 1) : 0.0,  (1, 1) : 1.0})),
         BNNode('T', ['R', 'S'],
                P(['T', 'R', 'S'], 
                   {(0, 0, 0) : 1.0, (1, 0, 0) : 0.0,
                   (0, 0, 1) : 0.1, (1, 0, 1) : 0.9,
                   (0, 1, 0) : 0.0, (1, 1, 0) : 1.0,
                   (0, 1, 1) : 0.0, (1, 1, 1) : 1.0}))])

# Test BN query method using the wet grass model.
def test2():
    print 'Testing prob'

    print "wg.prob({'R' : 1, 'S' : 1, 'T' : 0, 'J' : 0})"
    print wg.prob({'R' : 1, 'S' : 1, 'T' : 0, 'J' : 0})
    print "wg.prob({'R' : 0, 'S' : 0, 'T' : 0, 'J' : 0})"
    print wg.prob({'R' : 0, 'S' : 0, 'T' : 0, 'J' : 0})
    print "wg.prob({'R' : 1, 'S' : 0, 'T' : 0, 'J' : 0})"
    print wg.prob({'R' : 1, 'S' : 0, 'T' : 0, 'J' : 0})
    print "wg.prob({'R' : 0, 'S' : 1, 'T' : 0, 'J' : 0})"
    print wg.prob({'R' : 0, 'S' : 1, 'T' : 0, 'J' : 0})

    
    print 'Testing query'

    print "wg.query(['S'])"
    print wg.query(['S'])
    print "wg.query(['S'], ['T'], [1])"
    print wg.query(['S'], ['T'], [1])
    print "wg.query(['S'], ['T', 'J'], [1, 1])"
    print wg.query(['S'], ['T', 'J'], [1, 1])

    print "wg.query('R')"
    print wg.query('R')
    print "wg.query('R', ['T'], [1])"
    print wg.query('R', ['T'], [1])
    print "wg.query('R', ['T', 'S'], [1, 1])"
    print wg.query('R', ['T', 'S'], [1, 1])

print "Loaded gm.py"        
        

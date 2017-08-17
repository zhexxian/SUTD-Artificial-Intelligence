from search import *
from highways import *
import time
from itertools import islice

# An undirected graph of highways in USA.  The cost is defined using
# the distance function from highways.py.  The neighbors dictionary is
# also defined in highways.py. Make sure that you have downloaded the
# maps.zip file and placed it in the same directory as this file.

# NOTE THAT CREATING THIS GRAPH TAKES A BIT OF TIME, so do it only
# once if already defined in the environment.

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

try:
    # See if it is defined
    usa
except:
    # If it isn't, evaluate it
    usa = UndirectedGraph({id1:{id2:distance(id1,id2) for id2 in neighbors[id1]} \
                           for id1 in neighbors})
    
### Get more info about the database ###

    #print(type(location_from_ID)) #<type 'dict'>
    #print(take(5, location_from_names.iteritems()))
    '''
    [('CAPEBBLE BCH E  V', <highways.Location instance at 0x0329ED28>), 
    ('CAWINTERS   -WSW', <highways.Location instance at 0x07C11A80>), 
    ('AZDOUGLAS', <highways.Location instance at 0x032C5DC8>), 
    ('ID_LOST TRAIL PASS', <highways.Location instance at 0x032C5440>), 
    ('PAERIE CHARTER OAK', <highways.Location instance at 0x032C5DA0>)]
    '''
    #print(len(location_from_ID)) #90415
    #print(len(location_from_name)) #22328 unique location
    #print(len(links)) #125302

    #print(location_from_name['ILPEORIA CBD']) #<highways.Location instance at 0x08164A30>
    #print(location_from_name['ILPEORIA CBD'].id_number) #17001455

    #print(location_from_ID[8000302]) #<highways.Location instance at 0x16FF7E90>
    #print(location_from_ID[8000302].name) #COBOULDER E

    #print(neighbors[23000331]) #[23000323, 23000326, 23000333]

try:
    usa_graph_problem
except:
    usa_graph_problem = GraphProblem(20000071,25000502,usa)



class MyFIFOQueue(FIFOQueue):
    def getNode(self, state):
        '''Returns node in queue with matching state'''
        for i in range(self.start, len(self.A)):
            if self.A[i].state == state:
                return self.A[i]
    def __contains__(self, node):
        '''Returns boolean if there is node in queue with matching
        state.  The implementation in utils.py is very slow.'''
        for i in range(self.start, len(self.A)):
            if self.A[i].state == node.state:
                return True

def bidirectional_search(problem):
    '''
    Perform bidirectional search, both directions as breadth-first
    search, should return either the final (goal) node if a path is
    found or None if no path is found.
    '''
    assert problem.goal                # a fixed goal state

    # Below is the definition of BREADTH_FIRST_SEARCH from search.py.
    # You will need to (a) UNDERSTAND and (b) MODIFY this to do
    # bidirectional search.

    node_forward = Node(problem.initial)
    node_backward = Node(problem.goal)
    if problem.goal_test(node_forward.state):
        return node_forward
    frontier_forward = MyFIFOQueue()
    frontier_backward = MyFIFOQueue()
    frontier_forward.append(node_forward)
    frontier_backward.append(node_backward)
    explored_forward = set()
    explored_backward = set()

    while frontier_forward and frontier_backward:
        # forward search
        node_forward = frontier_forward.pop()
        explored_forward.add(node_forward.state)
        for child in node_forward.expand(problem):
            if child.state not in explored_forward and child not in frontier_forward:
                if child in frontier_backward:
                    '''
                    backward_child = frontier_backward.getNode(child.state)
                    #child.finalpath = child.path() + backward_child.path()
                    '''
                    forward_path = child.path()
                    backward_child = frontier_backward.getNode(child.state)

                    backward_path = backward_child.path()
                    node = backward_path.pop()

                    counter = len(backward_path)

                    while counter > 0:
                        parent = node
                        node = backward_path.pop()
                        node.parent = parent

                    return node

                frontier_forward.append(child)
         
        # backward search
        node_backward = frontier_backward.pop()
        explored_backward.add(node_backward.state)
        for child in node_backward.expand(problem):
            if child.state not in explored_backward and child not in frontier_backward:
                if child in frontier_forward: 
                    '''
                    forward_child = frontier_forward.getNode(child.state)
                    #child.finalpath = forward_child.path() + list(reversed(child.path()))
                    '''
                    forward_child = frontier_forward.getNode(child.state)
                    forward_path = forward_child.path()

                    backward_path = child.path()
                    node = forward_path.pop() #child

                    counter = len(backward_path)

                    while counter > 0:
                        parent = node
                        node = backward_path.pop()
                        node.parent = parent
                        counter -= 1

                    return node
                
                frontier_backward.append(child)
    return None

# Modified from search.py
def compare_searchers(problems, header,
                      h = None,
                      searchers=[breadth_first_search]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        print 'Starting', name(searcher)
        t0 = time.time()
        if name(searcher) in ('astar_search', 'greedy_best_first_graph_search'):
            searcher(p, h)
        else:
            searcher(p)
        t1 = time.time()
        print 'Completed', name(searcher)
        return p, t1-t0
    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)

def test_map():
    heuristic = lambda x: distance(x.state, 25000502)
    compare_searchers(problems = [GraphProblem(20000071, 25000502, usa)],
                      h = heuristic,
                      searchers = [breadth_first_search,
                                   bidirectional_search,
                                   uniform_cost_search,
                                   astar_search],
                      header = ['Searcher', 'USA(Smith Center, Cambridge)'])

def search_path(search):
    path = []
    for node in search.path():
        path.append(node.state)
    return path

def bidirectional_search_path(search):
    path = []
    for node in search.finalpath:
        path.append(node.state)
    return path

def path_cost(path):
    cost = 0
    for i in range(len(path)-1):
        cost = cost + distance(path[i],path[i+1])
    return cost

if __name__=='__main__':
    #to_kml(search_path(uniform_cost_search(usa_graph_problem)))
    #print(path_cost(search_path(uniform_cost_search(usa_graph_problem))))
    #print(bidirectional_search(usa_graph_problem))
    #to_kml(search_path(bidirectional_search(usa_graph_problem)))
    to_kml(search_path(bidirectional_search(usa_graph_problem)))
    #test_map()


    

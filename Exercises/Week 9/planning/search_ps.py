# Written by Patricia Suriana, MIT ca. 2013

from dataStructure import *
from heuristic import *


##################################################################################################
# Public Interface
##################################################################################################


def format_string(fact):
    name, args = fact[0], fact[1:]
    args_string = ' ' + ' '.join(args) if args else ''
    return '(%s%s)' % (name, args_string)

def aStarSearch(task, heuristic, useHelpfulAction=False, noDel=False):
    extendedPathsCount = 0

    root = SearchNode(task.initial_state, None, None, 0)
    open_set = PriorityQueue()
    open_set.push(root, 0)

    state_cost = {task.initial_state: 0}

    while not open_set.isEmpty():
        pop_node = open_set.pop()
        pop_state = pop_node.state

        # Extend only if the cost of the node is the cheapest found so far.
        # Otherwise ignore, since we've found a better one. 
        if state_cost[pop_state] == pop_node.cost:
            extendedPathsCount += 1

            if task.goal_reached(pop_state):
                print "Finished searching. Found a solution. "
                return (extendedPathsCount, pop_node.cost, pop_node.path(), pop_state)

            relaxedPlan = None
            if useHelpfulAction:
                relaxedPlan = heuristic.getRelaxedPlan(SearchNode(pop_state, None, None, 0))

            for op, succ_state in task.get_successor_states(pop_state, noDel):
                # If we're using helpful actions, ignore op that is not in the helpful actions
                if useHelpfulAction:
                    if relaxedPlan and not op.name in relaxedPlan:
                        #print str(op.name) + " not in " +  str(relaxedPlan)
                        continue

                # Assume each action (operation) has cost of one
                succ_node = SearchNode(succ_state, pop_node, op, 1)
                h = heuristic(succ_node)
                #print "\nHeuristic values for " + ', '.join(map(format_string, succ_node.state)) + " is " + str(h)

                if h == float('inf'):
                    # Don't bother with states that can't reach the goal anyway
                    continue

                old_succ_cost = state_cost.get(succ_state, float("inf"))
                if succ_node.cost < old_succ_cost:
                    # Found a cheaper state or never saw this state before
                    open_set.push(succ_node, succ_node.cost + h)
                    state_cost[succ_state] = succ_node.cost

    print "No more operations left. Cannot solve the task"
    # (extendedPathsCount, cost, path, final_state)
    return (extendedPathsCount, None, None, None)
    

# Written by Patricia Suriana, MIT ca. 2013
# Modified by Tomas Lozano-Perez, MIT ca 2016

import pddl_parser
import search
from search import Problem
from search import astar_search
from search import breadth_first_tree_search
import time
import sys
import pdb
from strips import Task
from strips import Operator

class PlanProblem(Problem):
    def __init__(self, name, facts, initial, goal, operators, noDel = False):
        self.name = name
        self.facts = facts
        self.initial = initial
        self.goal = goal
        self.operators = operators
        self.satisfied_count = 0
    def get_successor_states(self, state, noDel = False):
        return [(op, op.apply(state, noDel)) for op in self.operators
                if op.applicable(state)]
    def get_successor_ops(self, state):
        return [op for op in self.operators if op.applicable(state)]
    def actions(self, state):
        return self.get_successor_ops(state)
    def result(self, state, action):
        states = self.get_successor_states(state)
        for i in range(len(states)):
            if(states[i][0] == action):
                return states[i][1]
        return None
    def goal_reached(self, state):
        self.satisfied_count = len(state.difference(self.goal)) 
        self.h_add = len(state.difference(self.facts)) 
        return self.goal <= state
    def goal_test(self, state):
        return self.goal_reached(state)
    def value(self, state):
        pass
    def h(self, n):
        #return 0
        #return self.satisfied_count #h_g: count the number of unsatisfied goals
        return self.h_add #h_add

def printOutputVerbose(tic, toc, path, cost, final_state, goal):
    print "\n******************************FINISHED TEST******************************"
    print "Goals: "
    for state in goal:
        print "\t" + str(state)
    print '\nRunning time: ', (toc-tic), 's'
    if path == None:
        print '\tNO PATH FOUND'
    else:
        print "\nNumber of Actions: ", len(path)
        print '\nCost:', cost
        print "\nPath: "
        for op in path:
            print "\t" + repr(op)
        print "\nFinal States:"
        for state in final_state:
            print "\t" + str(state)
    print "*************************************************************************\n"

def printOutput(tic, toc, path, cost):
    print (toc-tic), '\t', len(path), '\t', cost          

if __name__ == "__main__":
  args = sys.argv
  if len(args) != 3:                    # default task
      #dirName = "prodigy-bw"
      #fileName = "bw-simple"
      #fileName = "bw-12step"
      dirName = "painting"
      #fileName = "p0"
      fileName = "p1"
  else:
      dirName = args[1]
      fileName = args[2]
  domain_file = dirName + '/domain.pddl'
  problem_file = dirName + '/' + fileName + '.pddl'

  # task is an instance of the Task class defined in strips.py
  task = pddl_parser.parse(domain_file, problem_file)

  # This should be commented out for larger tasks
  print task

  print "\n******************************START TEST******************************"
  
  tic = time.time()

  # Define a sub-class of the Problem class, make an instance for the task and call 
  # the search
  # You should then set the variables:
  # final_state - the final state at the end of the plan
  # plan - a list of actions representing the plan
  # cost - the cost of the plan
  # Your code here
  plan_problem = PlanProblem(name=task.name, facts=task.facts, 
    initial=task.initial_state, goal=task.goals, operators=task.operators)
  plan = astar_search(plan_problem).solution()
  cost = 0 #for simplicity
  final_state = astar_search(plan_problem).state
  toc = time.time()
  printOutputVerbose(tic, toc, plan, cost, final_state, task.goals)

  

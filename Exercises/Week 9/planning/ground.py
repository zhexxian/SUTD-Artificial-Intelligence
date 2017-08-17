# Written by Patricia Suriana, MIT ca. 2013
# Modified by Tomas Lozano-Perez, MII ca. 2016

import itertools
from strips import Task, Operator
from operator import itemgetter
import pdb

def ground(domain, problem):
    [predicates, actions] = domain
    [name, objects, init, goals] = problem

    # Get static predicates (predicates that are not in any of the 
    # operators' add_effects
    statics = get_static_predicates(predicates, actions)

    # Ground the actions. Get all possible grounded actions
    operators = ground_actions(actions, objects, statics, init)

    # Get all possible facts (for heuristic calculation during Dijkstra)
    facts = collect_facts(operators) | goals

    # Remove all static facts from init 
    init &= facts

    #print "\nBefore relevance analysis: size " + str(len(operators))
    
    # Get rid of useless ops. Keep only the relevant ops (the ones that
    # potentially will lead us to the goals)
    operators = get_relevant_operators(operators, goals)

    #print "\nAfter relevance analysis: size " + str(len(operators))
    return Task(name, facts, init, goals, operators)


def get_relevant_operators(operators, goals):
    '''
    Determine the operations the effects contribute to the 
    goal (have something to do to achieve the goal)
    '''
    old_relevant_facts = set()
    relevant_facts = set()
    for goal in goals:
        relevant_facts.add(goal)

    while True:
        old_relevant_facts = relevant_facts.copy()

        # Check if the operator's effects are relevant in 
        # achieving the goal
        for op in operators:
            del_intersect = op.del_effects & relevant_facts
            add_intersect = op.add_effects & relevant_facts
            # The operator is relevant; we want to make sure
            # we can achieve this operator
            if del_intersect or add_intersect:
                relevant_facts |= op.preconditions
        # It does not change anymore
        if old_relevant_facts == relevant_facts:
            break

    useless_ops = set()
    for op in operators:
        del_intersect = op.del_effects & relevant_facts
        add_intersect = op.add_effects & relevant_facts
        # Useless operator
        if not del_intersect and not add_intersect:
            useless_ops.add(op)
    # Remove useless operators
    return [op for op in operators if not op in useless_ops]


def collect_facts(operators):
    '''
    Collect all valid facts from grounded operators (precondition, add
    effects and delete effects).
    '''
    facts = set()
    for op in operators:
        facts |= op.preconditions | op.add_effects | op.del_effects
    return facts


def ground_actions(actions, object_list, statics, init):
    op_lists = [ground_action(action, object_list, statics, init)
                for action in actions]
    operators = list(itertools.chain(*op_lists))
    return operators


def OLD_ground_action(action, object_list, statics, init):
    '''
    Ground the action and return list of possible operators (grounded
    actions).  
    '''
    param_to_objects = {}

    for param_name in action.parameters:
        param_to_objects[param_name] = object_list

    # Save a list of possible assignment tuples (param_name, object)
    domain_lists = [[(name, obj) for obj in objects] for name, objects in
                    param_to_objects.items()]
    # Calculate all possible assignments
    assignments = itertools.product(*domain_lists)
    # Create a new operator for each possible assignment of parameters
    ops = [create_operator(action, dict(assign), statics, init)
            for assign in assignments]
    # Filter out the None values
    ops = filter(bool, ops)
    return ops


def ground_action(action, object_list, statics, init):
    '''
    Ground the action and return list of possible operators (grounded
    actions).  The assignments to operator variables are the solutions
    of a CSP, where the constraints are the static preconditions.
    '''
    static_preconds = []
    static_unary_preconds_by_param = {param_name : [] for param_name in action.parameters}
    for precondition in action.precondition:
        predicate_name = precondition[0]
        if predicate_name in statics:
            static_preconds.append(precondition)
            if len(precondition) == 2:
                static_unary_preconds_by_param[precondition[1]].append(precondition)

    param_to_objects = {}
    for param_name in action.parameters:
        relevant_objects = filter_objects(object_list, static_unary_preconds_by_param[param_name], init)
        param_to_objects[param_name] = relevant_objects

    def consistent(assign):
        for precond in static_preconds:
            params = precond[1:]
            if all(p in assign for p in params):
                if not tuple([precond[0]] + [assign[p] for p in params]) in init:
                    return False
        return True

    assignments = backtrack_assignments(param_to_objects, consistent)
    print action.name, 'has', len(assignments), 'instances'

    # Create a new operator for each possible assignment of parameters
    ops = [create_operator(action, assign, statics, init)
            for assign in assignments]
    # Filter out the None values
    ops = filter(bool, ops)
    return ops

def filter_objects(object_list, static_unary_preconds, init):
    '''
    Return a list of objects consistent with facts in init.
    '''
    if not static_unary_preconds:
        return object_list
    return [obj for obj in object_list \
            if all((p[0], obj) in init for p in static_unary_preconds)]

# Return all the solutions (up to maxSol).
def backtrack_assignments(param_to_objects, consistent):
    # sort variables by size of domain
    p_d_list = sorted(param_to_objects.items(), key=lambda p_d: len(p_d[1]))
    variables = [p_d[0] for p_d in p_d_list]
    domains = param_to_objects
    answers = []
    def backtrack_rec(i, var, j, varValues, assignment):
        if j >= len(varValues):
            return
        newAssignment = assignment.copy()
        newAssignment[var] = varValues[j]
        if consistent(newAssignment):
            if len(newAssignment) == len(domains):
                answers.append(newAssignment)
            else:
                backtrack_rec(i+1, variables[i+1], 0, domains[variables[i+1]],
                             newAssignment)
            backtrack_rec(i, var, j+1, varValues, assignment)
        else:
            backtrack_rec(i, var, j+1, varValues, assignment)
    backtrack_rec(0, variables[0], 0, domains[variables[0]], {})
    return answers

def get_static_predicates(predicates, actions):
    '''
    Determine all static predicates and return them as a list. A static 
    predicate does not occur in the effects of the action
    '''
    def get_effects(action):
        return action.add_effects | action.del_effects

    effects = [get_effects(action) for action in actions]
    effects = set(itertools.chain(*effects))

    def static(predicate):
        return not any(predicate.name == eff[0] for eff in effects)

    statics = [pred.name for pred in predicates if static(pred)]
    return statics


def create_operator(action, assignment, statics, init):
    '''
    Given an assignment, ground the action
    '''
    precondition_facts = set()
    for precondition in action.precondition:
        fact = ground_precondition(precondition, assignment)
        predicate_name = precondition[0]
        if predicate_name in statics:
            # Check if this precondition is false in the initial state
            if fact not in init:
                # This precondition is never true -> Don't add operator
                return None
        else:
            # This precondition is not always true -> Add it
            precondition_facts.add(fact)

    add_effects = action.ground_add_effects(assignment)
    del_effects = action.ground_del_effects(assignment)
    # If the same fact is added and deleted by an operator the STRIPS formalism
    # adds it.
    del_effects -= add_effects
    # If a fact is present in the precondition, we do not have to add it.
    # Note that if a fact is in the delete and in the add effects,
    # it has already been deleted in the previous step.
    add_effects -= precondition_facts

    def format_string(name, args):
        args_string = ' ' + ' '.join(args) if args else ''
        return '(%s%s)' % (name, args_string)

    args = [assignment[name] for name in action.parameters]
    name = format_string(action.name, args)
    op = Operator(name, precondition_facts, add_effects, del_effects)
    return op


def ground_precondition(precondition, assignment):
    '''
    Return a string representing the grounded precondition with
    respect to the assignment
    '''
    result = [precondition[0]]
    for arg in precondition[1:]:
        assert arg in assignment
        result.append(assignment[arg])
    return tuple(result)

# Written by Patricia Suriana, MIT ca. 2013

import pddl_types
from copy import deepcopy

class Action(object):
    def __init__(self, name, parameters, precondition, add_effects, del_effects):
        self.name = name
        # :parameters (?stackObj ?onObj)
        self.parameters = parameters
        self.precondition = frozenset(precondition)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)
        
    def dump(self):
        print "Action: " + self.name
        print "Parameters:"
        for e in self.parameters:
            print '\t' + str(e) + '\n'
        print "Preconditions:"
        for e in self.precondition:
            print '\t' + str(e) + '\n'
        print "Add effects:"
        for e in self.add_effects:
            print '\t' + str(e) + '\n'
        print "Delete effects:"
        for e in self.del_effects:
            print '\t' + str(e) + '\n'

    def ground(self, ground_map, array):
        # Format: (type, arg1, arg2, ...)
        assertion = []
        for el in array:
            entry = [el[0]]
            for arg in el[1:]:
                assert arg in ground_map
                entry.append(ground_map[arg])
            assertion.append(tuple(entry))
        return frozenset(assertion)

    def ground_add_effects(self, ground_map):
        return self.ground(ground_map, self.add_effects)

    def ground_del_effects(self, ground_map):
        return self.ground(ground_map, self.del_effects)

    def __str__(self):
        return "<Action %r at %#x>" % (self.name, id(self))
    __repr__ = __str__
    
    
'''
(:action pick-up
    :parameters (?obj)
    :precondition (and (clear ?obj) (on-table ?obj) (arm-empty))
    :effect
        (and (not (on-table ?obj))
            (not (clear ?obj))
            (not (arm-empty))
            (holding ?obj)
        )
)
'''
def parse_action(alist):
    iterator = iter(alist)
    assert iterator.next() == ":action"
    name = iterator.next()
    parameters_tag_opt = iterator.next()
    if parameters_tag_opt == ":parameters":
        parameters = pddl_types.parse_typed_object_list(iterator.next(),
                                                        variable_names=True)
        precondition_tag_opt = iterator.next()
    else:
        parameters = []
        precondition_tag_opt = parameters_tag_opt

    if precondition_tag_opt == ":precondition":
        precondition = parse_precondition(iterator.next())
        effect_tag = iterator.next()
    else:
        precondition = []
        effect_tag = precondition_tag_opt

    assert effect_tag == ":effect"
    effect_list = iterator.next()
    add_effects = []
    del_effects = []

    parse_effect_list(effect_list, add_effects, del_effects)
    for rest in iterator:
        assert False, rest
    return Action(name, parameters, precondition, add_effects, del_effects)


'''
(:effect
    (and (not (on-table ?obj))
        (not (clear ?obj))
        (not (arm-empty))
        (holding ?obj)
    )
)
'''
def parse_effect_list(effect_list, add_effects, del_effects):
    if effect_list[0] != "and":
        entry = effect_list
        parse_effect(entry, add_effects, del_effects)
        return 

    for entry in effect_list[1:]:
        parse_effect(entry, add_effects, del_effects)

def parse_effect(effect, add_effects, del_effects):
    if effect[0] == "not": # Delete effect
        assert len(effect[1:]) == 1
        del_effects.append(tuple(effect[1]))
    else: 
        add_effects.append(tuple(effect))

# :precondition (and (clear ?obj) (on-table ?obj) (arm-empty))
def parse_precondition(alist):
    precondition = []
    if alist[0] != "and":
        # assert len(alist) == 1
        entry = alist
        precondition.append(tuple(entry))
        return precondition

    assert len(alist[1:]) >= 1       
    for entry in alist[1:]:
        precondition.append(tuple(entry))
    return precondition


if __name__ == "__main__":
    action = [':action', 'paint', ':parameters', ['?origcol', '?newcol', '?paintingtool', '?obj'], \
              ':precondition', ['and', ['on-table', '?obj'], ['clear', '?obj'], ['colored', '?obj', '?origcol'], \
              ['holding', '?paintingtool'], ['paintingtool', '?paintingtool'], ['has-paint-color', '?paintingtool', \
              '?newcol']], ':effect', ['and', ['colored', '?obj', '?newcol'], ['not', ['colored', '?obj', '?origcol']]]] 
    result = parse_action(action)
    #result.dump()

    #ground_map = result.createGroundMap(object_map)

    ground_map = dict([('?obj1', 'A'), ('?obj2', 'B'), ('?brush', 'brush'),\
                       ('?paint', 'paint'), ('?obj', 'C'), ('?origcol', 'red'),\
                       ('?newcol', 'blue'), ('?paintingtool', 'spray')])

    print "Grounded Preconditions:"
    for e in result.ground_precondition(ground_map):
        print '\t' + str(e) + '\n'

    print "Grounded Add Effects:"
    for e in result.ground_add_effects(ground_map):
        print '\t' + str(e) + '\n'

    print "Grounded Delete Effects:"
    for e in result.ground_del_effects(ground_map):
        print '\t' + str(e) + '\n'

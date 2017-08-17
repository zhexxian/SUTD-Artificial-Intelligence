# Written by Patricia Suriana, MIT ca. 2013

import pddl_types

'''
self.name is name of the predicate
self.arguments is a list of TypedObjects which are arguments of the
predicate
'''
class Predicate(object):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
    def __str__(self):
        return "%s(%s)" % (self.name, ", ".join(map(str, self.arguments)))
    def __repr__(self):
        return self.__str__()

    def uniquify_variables(self, ground_map):
        # Format: (type, arg1, arg2, ...)
        assertion = [self.name]
        for arg in self.arguments:
            assert arg in ground_map
            assertion.append(ground_map[arg])
        return tuple(assertion)

# Example of alist: [on ?obj1 ?obj2] or [arm-empty] or [colored ?obj ?color]
def parse_predicate(alist):
    name = alist[0]
    arguments = pddl_types.parse_typed_object_list(alist[1:], variable_names=True)
    return Predicate(name, arguments)
    

if __name__ == "__main__":
    pred_list = [['on', '?obj1', '?obj2'], ['on-table', '?obj'], ['clear', '?obj'], \
    			 ['arm-empty'], ['holding', '?obj'], ['colored', '?obj', '?color'], \
    			 ['clean', '?brush'], ['paintingtool', '?obj'], ['has-paint-color', '?obj', '?paint']]
    result = [parse_predicate(entry) for entry in pred_list]
    for e in result:
        print e, '\n'

    ground_map = dict([('?obj1','A'),('?obj2','B'),('?brush','brush'),\
                       ('?paint','paint'),('?obj','C'),('?color','red')])
    for e in result:
        print e.uniquify_variables(ground_map), '\n'

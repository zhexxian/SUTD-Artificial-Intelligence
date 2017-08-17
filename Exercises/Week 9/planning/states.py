# Written by Patricia Suriana, MIT ca. 2013

'''
Set of propositions (assertions) each of which is a tuple of the 
form (type, arg1, arg2, ...), for example, ('free', 'partA')
'''

'''
alist = [(arm-empty), (on-table A), (and (arm-empty) (colored A red))]
'''
def parse_state_list(alist):
    assertions = parse_state_list_helper(alist)
    return frozenset(assertions)

def parse_state_list_helper(alist):
    assertions = []
    for state in alist:
        if state != "and":
            entry = state
            assertions.append(tuple(state))
        else:
            continue
    return assertions


if __name__ == "__main__":
    state_list = [['arm-empty'], ['on-table', 'a'], ['clear', 'a'], \
                  ['colored', 'a', 'none'], ['paintingtool', 'redsprayer'], \
                  ['has-paint-color', 'redsprayer', 'red'], ['on-table', 'redsprayer'], \
                  ['clear', 'redsprayer'], ['and', ['arm-empty'], ['colored', 'a', 'red']]]
    result = parse_state_list(state_list)
    for e in result.assertions:
        print e, '\n'

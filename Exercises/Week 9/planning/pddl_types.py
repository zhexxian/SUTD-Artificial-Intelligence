# Written by Patricia Suriana, MIT ca. 2013

# (:types BLOCK PAINTSPRAYER PAINTCAN PAINTBRUSH WATERBUCKET COLOR)
def parse_type_list(alist):
    result = []
    for typeName in alist:
        entry = Type(typeName, "object")
        result.append(entry)
    return result

'''
some predicate's arguments: (on ?obj1 ?obj2), etc
'''
def parse_typed_object_list(alist, variable_names=False, makeDict=False):
    result = []
    while alist:
        items = alist
        alist = []
        for item in items:
            # If it is an argument to a predicate, it should start
            # with a "?"
            assert not variable_names or item.startswith("?")
            result.append(item)
    return result

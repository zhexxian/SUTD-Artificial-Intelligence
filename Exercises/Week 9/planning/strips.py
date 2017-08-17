# Written by Patricia Suriana, MIT ca. 2013

class Operator:
    '''
    The preconditions represent the facts that have to be true
    before the operator can be applied.
    add_effects are the facts that the operator makes true.
    delete_effects are the facts that the operator makes false.
    '''
    def __init__(self, name, preconditions, add_effects, del_effects):
        self.name = name
        self.preconditions = frozenset(preconditions)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)

    def applicable(self, state):
        '''
        True if the preconditions is subset of state
        '''
        return self.preconditions <= state

    def apply(self, state, noDel = False):
        assert self.applicable(state)
        assert type(state) in (frozenset, set)
        if noDel:
            return state | self.add_effects
        return (state - self.del_effects) | self.add_effects

    def __str__(self):
        s = '%s\n' % self.name
        for group, facts in [('PRE', self.preconditions),
                             ('ADD', self.add_effects),
                             ('DEL', self.del_effects)]:
            for fact in facts:
                s += '  %s: %s\n' % (group, fact)
        return s

    def __repr__(self):
        return '<Op %s>' % self.name


class Task:
    '''
    Represents a particular problem to solve (name, facts,
    initial_state, goals and operators).
    '''
    def __init__(self, name, facts, initial_state, goals, operators):
        self.name = name
        self.facts = facts
        self.initial_state = initial_state
        self.goals = goals
        self.operators = operators

    def goal_reached(self, state):
        '''
        If all assertions within self.goals is included in state, return
        True; otherwise false
        '''
        return self.goals <= state

    def get_successor_states(self, state, noDel = False):
        '''
        Get a list of successor states of the form (op, new_state) where "op" is the
        operator applied to achieve the new state "new_state". If noDel is True, 
        we use no-delete list where delete effects are ignored. 
        '''
        return [(op, op.apply(state, noDel)) for op in self.operators
                if op.applicable(state)]

    def get_successor_ops(self, state):
        '''
        Get a list of ops that are applicable to the current state
        '''
        return [op for op in self.operators if op.applicable(state)]

    def __str__(self):
        def format_string(fact):
            name, args = fact[0], fact[1:]
            args_string = ' ' + ' '.join(args) if args else ''
            return '(%s%s)' % (name, args_string)

        s = 'Task {0}\n\n  Facts:  {1}\n\n  Init:  {2}\n\n  Goals: {3}\n\n  Ops:   {4}'
        return s.format(self.name, ', '.join(map(format_string, self.facts)),
                            ', '.join(map(format_string, self.initial_state)), 
                            ', '.join(map(format_string, self.goals)),
                            '\n'.join(map(repr, self.operators)))

    def __repr__(self):
        string = '<Task {0}, vars: {1}, operators: {2}>'
        return string.format(self.name, len(self.facts), len(self.operators))



# Written by Patricia Suriana, MIT ca. 2013

import tasks
import ground

__all__ = ["ParseError", "parse"]

class ParseError(Exception):
  pass


def parse(domain_file, problem_file):
  domain_line = parse_nested_list(file(domain_file))
  problem_line = parse_nested_list(file(problem_file))

  domain = tasks.parse_domain(domain_line)
  problem = tasks.parse_task(problem_line)
  
  task = ground.ground(domain, problem)
  return task


# Basic functions for parsing PDDL files.
def parse_nested_list(input_file):
  tokens = tokenize(input_file)
  next_token = tokens.next()
  if next_token != "(":
    raise ParseError("Expected '(', got %s." % next_token)
  result = list(parse_list_helper(tokens))
  for tok in tokens:  # Check that generator is exhausted.
    raise ParseError("Unexpected token: %s." % tok)
  return result
  

def tokenize(input):
  for line in input:
    line = line.split(";", 1)[0]  # Strip comments.
    line = line.replace("(", " ( ").replace(")", " ) ").replace("?", " ?")
    for token in line.split():
      yield token.lower()


def parse_list_helper(tokenstream):
  # Leading "(" has already been swallowed.
  while True:
    try:
      token = tokenstream.next()
    except StopIteration:
      raise ParseError()
    if token == ")":
      return
    elif token == "(":
      yield list(parse_list_helper(tokenstream))
    else:
      yield token
      

if __name__ == "__main__":
  import strips
  import ground
  
  domain_file = 'domain.pddl'
  task_file = 'p0.pddl'
  domain_line = parse_nested_list(file(domain_file))
  task_line = parse_nested_list(file(task_file))
  domain = tasks.parse_domain(domain_line)
  predicates, actions = domain
  task = tasks.parse_task(task_line)
  [problem_name, objects, init, goal] = task

  print "Problem Name: " + problem_name

  #print "Predicates:"
  #for e in predicates:
  #  print '\t', e, '\n'
  #print "Actions:"
  #for e in actions:
  #  print '\t', e, '\n'

  statics = ground.get_static_predicates(predicates, actions)
  #print "Statics Predicate:"
  #for e in statics:
  #  print '\t', e, '\n'

  assignment = {'?obj': 'blockA', '?newcol': 'red', '?origcol': 'none', \
                '?paintingtool': 'redsprayer'}
  #op = ground.create_operator(actions[4], assignment, statics, init)
  #print op

  grounded = ground.ground_action(actions[4], objects, statics, init)
  for e in grounded:
    print '\t', e, '\n'


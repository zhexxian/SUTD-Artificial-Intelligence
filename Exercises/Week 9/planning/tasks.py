# Written by Patricia Suriana, MIT ca. 2013

import actions
import predicates
import pddl_types
import states


'''
Only support :predicates and :action (no :type or :constant)
Output: predicate, action
'''
def parse_domain(domain_pddl):
  iterator = iter(domain_pddl)

  assert iterator.next() == "define"
  domain_line = iterator.next()
  assert domain_line[0] == "domain" and len(domain_line) == 2
  #yield domain_line[1]

  # Parse :requirements (We don't use this)
  requirements = iterator.next()
  assert requirements[0] == ":requirements"
  predicate_line = iterator.next()
  
  # Parse :predicates
  predicate_list = []
  assert predicate_line[0] == ":predicates" and len(predicate_line) >= 2
  for entry in predicate_line[1:]:
    predicate_list.append(predicates.parse_predicate(entry))
  
  # Parse :action
  action_list = []
  first_action = iterator.next()
  assert first_action[0] == ":action"
  entries = [first_action] + [entry for entry in iterator]
  for entry in entries:
    action = actions.parse_action(entry)
    action_list.append(action)

  return [predicate_list, action_list]

'''
Output objects, init, and goal
'''
def parse_task(task_pddl):
  iterator = iter(task_pddl)

  assert iterator.next() == "define"
  problem_line = iterator.next()
  assert problem_line[0] == "problem" and len(problem_line) == 2
  problem_name = problem_line[1]
  domain_line = iterator.next()
  assert domain_line[0] == ":domain" and len(domain_line) == 2
  #yield domain_line[1]

  # Check if the :objects are specified
  objects_opt = iterator.next()
  if objects_opt[0] == ":objects":
    object_list = pddl_types.parse_typed_object_list(objects_opt[1:])
    init = iterator.next()
  else:
    object_list = []
    init = objects_opt

  # Parse :init
  assert init[0] == ":init" and len(init) > 1
  init_list = states.parse_state_list(init[1:])

  # Parse :goal
  goal = iterator.next()
  assert goal[0] == ":goal" and len(goal) == 2
  goal_list = states.parse_state_list(goal[1])

  return [problem_name, object_list, init_list, goal_list]

#! /bin/python3
import copy
from typing import List, Set, Tuple, Union
from ns_lib.logic.commons import Atom, Rule
from ns_lib.grounding.known_body_grounder import KnownBodyGrounder

# Performs forward chaining for a set of rules and given facts.
# The rules can be general clauses body -> head with at most two elements
# in the body.
class KnownBodyForwardGrounder(KnownBodyGrounder):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]]):
        super().__init__(rules, facts)

    def ground_one_rule(self, rule: Rule, queries: List[Tuple]):

      assert len(rule.body) <= 2, (
        'Clauses with up to 2 elements in the body are supported')

      new_groundings = set()

      for q in queries:
        for i,body_atom in enumerate(rule.body):
          if q[0] != body_atom[0]:
            continue

          # One ground atom is the query.
          ground_body_atom = q
          # print('ground_body_atom', q)
          # Get the variable assignments from the body.
          body_var_assignments = {v: a for v, a in zip(body_atom[1:],
                                                       ground_body_atom[1:])}

          if len(rule.body) > 1:
            # Select the other atom in the body and ground it with the
            # assignments with the head and the other body ground atom fixed.
            body_atom2 = rule.body[(i + 1) % 2]
            # print('body_atom2', body_atom2)
            # Partially ground the other atom with the variables that are
            # already determined.
            ground_body_atom2 = (body_atom2[0], ) + tuple(
                [body_var_assignments.get(body_atom2[j+1], None)
                 for j in range(len(body_atom2)-1)])
            # print('ground_body_atom2', ground_body_atom2)
            if all(ground_body_atom2[1:]):  # Fully ground, we are done
              groundings = (ground_body_atom2,)
            else:
              # Tuple of atoms matching A(Antonio,None) in the facts.
              # This is the list of ground atoms for the i-th atom.
              groundings = self._fact_index._index.get(ground_body_atom2, [])
            # print('groudings', groundings)
            for ground_atom in groundings:
              var_assignments = copy.copy(body_var_assignments)
              var_assignments.update(
                  {v: a for v, a in zip(body_atom2[1:], ground_atom[1:])})
              ground_head_atom_tuple = []
              all_ground_head = True
              for head in rule.head:
                  ground_head_atom = (head[0], ) + tuple(
                      [var_assignments.get(head[j+1], None)
                       for j in range(len(head)-1)])
                  if all(ground_head_atom):
                      ground_head_atom_tuple.append(ground_head_atom)
                  else:
                      all_ground_head = False
                      continue
              if all_ground_head:
                  new_body_grounding = (ground_body_atom, ground_atom)
                  new_groundings.add((tuple(ground_head_atom_tuple),
                                      new_body_grounding))
          else:
            var_assignments = copy.copy(body_var_assignments)
            ground_head_atom_tuple = []
            all_ground_head = True
            for head in rule.head:
                ground_head_atom = (head[0], ) + tuple(
                    [var_assignments.get(head[j+1], None)
                     for j in range(len(head)-1)])
                if all(ground_head_atom):
                    ground_head_atom_tuple.append(ground_head_atom)
                else:
                    all_ground_head = False
                    continue
            if all_ground_head:
                new_body_grounding = (ground_body_atom,)
                new_groundings.add((tuple(ground_head_atom_tuple),
                                    new_body_grounding))

      self.rule2groundings[rule.name].update(new_groundings)

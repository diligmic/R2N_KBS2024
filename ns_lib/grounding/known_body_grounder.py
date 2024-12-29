#! /bin/python3
import copy
from typing import Dict, List, Set, Tuple, Union
from ns_lib.logic.commons import Atom, Rule, RuleGroundings
from ns_lib.grounding.engine import Engine
from ns_lib.grounding.backward_chaining_grounder import AtomIndex
from itertools import product

class KnownBodyGrounder(Engine):

    def __init__(self, rules: List[Rule],
                 facts: List[Union[Atom, str, Tuple]]):
        self._name2rule = {rule.name: rule for rule in rules}
        self.rules = rules
        self.facts = [a if isinstance(a,Tuple) else
                      (a.toTuple() if isinstance(a,Atom) else
                      Atom(s=a).toTuple()) for a in facts]
        self._fact_index = AtomIndex(self.facts)


    def _init_internals(self, queries: List[Tuple]):
        self.relation2queries = {}
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = []
            self.relation2queries[q[0]].append(q)

        self.rule2groundings = {}
        for rule in self.rules:
            #if rule.name not in self.rule2groundings:
            self.rule2groundings[rule.name] = set()

    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []

        self._init_internals(queries)
        for rule in self.rules:
            rel_queries = self.relation2queries.get(rule.head[0][0], [])
            if rel_queries:
                if len(rule.body) <= 2:
                    self.ground_one_rule_body_len2(rule, rel_queries)
                else:
                    self.ground_one_rule(rule, rel_queries)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name:
                   RuleGroundings(rule_name, sorted(list(groundings),
                                                    key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name:
                   RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}
        # print('Queries', len(queries), 'Groundings', [len(v.groundings) for v in ret.values()], '=', sum([len(v.groundings) for v in ret.values()]))
        return ret

    # Special case for rules with body len < 2.
    def ground_one_rule_body_len2(
        self, rule: Rule, queries: List[Tuple]):
      # We have a rule like A(x,y) B(y,z) => C(x,z)
      assert len(rule.head) == 1, (
          'Rule is not a Horn clause %s' % str(rule))
      head = rule.head[0]

      assert len(rule.body) <= 2, (
          'Horn clauses with more than 2 elements in the body are '
          'not supported yet')

      new_groundings = set()

      for q in queries:
        if q[0] != head[0]:  # predicates must match.
          continue

        # Get the variable assignments from the head.
        head_var_assignments = {v: a for v, a in zip(head[1:], q[1:])}

        # Ground first body atom by replacing variables with constants.
        # The result is the partially ground atom A('Antonio',None)
        # with None indicating unground variables.
        body_atom = rule.body[0]
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_var_assignments.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        if all(ground_body_atom[1:]):
          # Variables all match, so we have already the wanted grounding.
          # Rule was in the form A(x,y) ^ ... -> B(x,y)
          groundings = (ground_body_atom,)
        else:
          # One varibale match, rule was in the form A(x,z) ^ ... -> B(x,y)
          # Tuple of atoms matching A(Antonio,None) in the facts.
          # This is the list of ground atoms for the i-th atom in the body.
          # groundings = self._fact_index.get_matching_atoms(ground_body_atom)
          groundings = self._fact_index._index.get(ground_body_atom, [])  # optimization to avoid one extra function call

        if len(rule.body) == 1:
          # Shortcut, we are done, the clause has no free variables.
          # Return the groundings.
          new_groundings.add(((q,), groundings))
          continue

        # Select the other atom in the body and ground it with the
        # assignments with the head and the other body ground atom fixed.
        body_atom2 = rule.body[1]

        for atom in groundings:
          head_body_var_assignments = copy.copy(head_var_assignments)
          head_body_var_assignments.update(
              {v: a for v, a in zip(body_atom[1:], atom[1:])})
          new_grounding = (body_atom2[0], ) + tuple(
              [head_body_var_assignments.get(body_atom2[j+1], None)
               for j in range(len(body_atom2)-1)])
          if all(new_grounding) and self._fact_index._index.get(
              new_grounding, []):
              # (body_atom1, body_atom2)
              body_grounding = (atom, new_grounding)
              new_groundings.add(((q,), body_grounding))
              # print('ADDED', q, '->', tuple(body_grounding))

      self.rule2groundings[rule.name].update(new_groundings)

    def ground_one_rule(self, rule: Rule, queries: List[Tuple]):
      # We have a rule like A(x,y) B(y,z) => C(x,z)
      assert len(rule.head) == 1, (
          'Rule is not a Horn clause %s' % str(rule))
      head = rule.head[0]

      new_groundings = set()

      for q in queries:
        if q[0] != head[0]:  # predicates must match.
          continue

        # Get the variable assignments from the head.
        head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}
        var2constants = {var:[c] for var,c in head_ground_vars.items()}
        for i,body_atom in enumerate(rule.body):
            # Ground first body atom by replacing variables with constants.
            # The result is the partially ground atom A('Antonio',None)
            # with None indicating unground variables.
            ground_body_atom = (body_atom[0], ) + tuple(
                [head_ground_vars.get(body_atom[j], None)
                 for j in range(1, len(body_atom))])
            # optimization to avoid one extra function call.
            atom_candidates = self._fact_index._index.get(ground_body_atom, [])
            for j in range(1, len(body_atom)):
                if ground_body_atom[j] is not None:
                    continue
                constant_candidates = [a[j] for a in atom_candidates]
                var = body_atom[j]
                if var in var2constants:
                    var2constants[var] = list(set(var2constants[var]) &
                                              set(constant_candidates))
                else:
                    var2constants[var] = list(set(constant_candidates))
                if len(var2constants[var]) == 0:
                    break

        vars = var2constants.keys()
        for ground_vars in product(*[c for c in var2constants.values()]):
            full_ground_vars = dict(zip(vars, ground_vars))
            ground_body_atoms = []
            for body_atom in rule.body:
                ground_body_atom = (body_atom[0], ) + tuple(
                    [full_ground_vars.get(body_atom[j+1], None)
                     for j in range(len(body_atom)-1)])
                ground_body_atoms.append(ground_body_atom)

            new_groundings.add(((q,), tuple(ground_body_atoms)))

      self.rule2groundings[rule.name].update(new_groundings)

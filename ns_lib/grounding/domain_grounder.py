from itertools import product
from typing import Dict, List, Set, Tuple, Union

from ns_lib.logic.commons import Rule,  Atom, Domain, RuleGroundings
from ns_lib.grounding.engine import Engine


class DomainFullGrounder(Engine):

    def __init__(self,
                 rules: List[Rule],
                 domains:Dict[str, Domain],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 pure_adaptive: bool=False,
                 exclude_symmetric: bool=False,
                 exclude_query: bool=False,
                 limit: int=None):

        self.rules = rules
        # The flat grounder is not query oriented.
        self.domains = domains
        self.domain2adaptive_constants = domain2adaptive_constants
        self.pure_adaptive = pure_adaptive
        if self.pure_adaptive:
            assert self.domain2adaptive_constants, (
                'Need adaptive variable if in pure adaptive mode.')
        self.limit = limit
        self.exclude_symmetric = exclude_symmetric
        self.exclude_query = exclude_query

    #@lru_cache
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        res = {}
        for rule in self.rules:
            added = 0
            groundings = []

            if self.pure_adaptive:
                ground_var_groups = [self.domain2adaptive_constants.get(d, [])
                                     for d in rule.vars2domain.values()]
            elif self.domain2adaptive_constants is not None:
                ground_var_groups = [self.domains[d].constants +
                                     self.domain2adaptive_constants.get(d, [])
                                     for d in rule.vars2domain.values()]
            else:
                ground_var_groups = [self.domains[d].constants
                                     for d in rule.vars2domain.values()]

            for ground_vars in product(*ground_var_groups):
                var_assignments = {k:v for k,v in zip(
                    rule.vars2domain.keys(), ground_vars)}

                is_good = True
                body_atoms = []
                for atom in rule.body:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    if (self.exclude_symmetric and
                        ground_atom[1] == ground_atom[2]):
                        is_good = False
                        break
                    if self.exclude_query and ground_atom in queries:
                        is_good = False
                        break
                    body_atoms.append(ground_atom)

                head_atoms = []
                for atom in rule.head:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], atom[j+1])
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    if self.exclude_symmetric and ground_atom[1] == ground_atom[2]:
                        is_good = False
                        break
                    if self.exclude_query and ground_atom in queries:
                        is_good = False
                        break
                    head_atoms.append(ground_atom)

                # Check that nothing has been discarded.
                if is_good:
                    groundings.append((tuple(head_atoms), tuple(body_atoms)))
                    added += 1
                    if self.limit is not None and self.limit >= added:
                        break

            res[rule.name] = RuleGroundings(rule.name, groundings=groundings)

        return res


# Like a DomainGrounder but only the body variables are domain grounded,
# while the head atoms are not expanded and set to the query atoms.
class DomainBodyFullGrounder(Engine):

    def __init__(self, rules: List[Rule],
                 domains:Dict[str, Domain],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 pure_adaptive: bool=False,
                 exclude_symmetric: bool=False,
                 exclude_query: bool=False,
                 limit: int=None):
        self.rules = rules
        self._name2rule = {rule.name: rule for rule in rules}
        # The flat grounder is not query oriented.
        self.domains = domains
        self.domain2adaptive_constants = domain2adaptive_constants
        self.pure_adaptive = pure_adaptive
        if self.pure_adaptive:
            assert self.domain2adaptive_constants, (
                'Need adaptive variable if in pure adaptive mode.')

        self.limit = limit
        self.exclude_symmetric = exclude_symmetric
        self.exclude_query = exclude_query

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

    #@lru_cache
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
                self.ground_one_rule(rule, rel_queries)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name:
                   RuleGroundings(rule_name,
                                  sorted(list(groundings),
                                         key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name:
                   RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}

        # print('GROUNDING DONE', sum([len(r.groundings) for r in ret]))
        return ret


    def ground_one_rule(self, rule: Rule, queries: List[Tuple]) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
      # We have a rule like A(x,y) ^ B(y,z) => C(x,z)
      assert len(rule.head) == 1, (
          'Rule is not a Horn clause %s' % str(rule))
      head = rule.head[0]

      new_groundings = set()

      for i,q in enumerate(queries):
        if q[0] != head[0]:  # predicates must match.
          continue

        added = 0
        # Get the variable assignments from the head.
        # If q = Married(Marco,Alice) and head = Married(x,y) then
        # head_var_assignments = {x: Marco, y=Alice}
        head_var_assignments = {v: a for v, a in zip(head[1:], q[1:])}

        # Iterate over variables that are not ground by the query.
        vars_to_ground = {v:d for v,d in rule.vars2domain.items()
                          if v not in head_var_assignments}

        if self.pure_adaptive:
            ground_var_groups = [self.domain2adaptive_constants.get(d, [])
                                 for d in vars_to_ground.values()]
        elif self.domain2adaptive_constants is not None:
            ground_var_groups = [self.domains[d].constants +
                                 self.domain2adaptive_constants.get(d, [])
                                 for d in vars_to_ground.values()]
        else:
            ground_var_groups = [self.domains[d].constants
                                 for d in vars_to_ground.values()]

        for ground_vars in product(*ground_var_groups):
            var_assignments = {k:v for k,v in zip(
                vars_to_ground.keys(), ground_vars)}
            var_assignments.update(head_var_assignments)

            body_atoms = []
            is_good = True
            for atom in rule.body:
                ground_atom = (atom[0], ) + tuple(
                    [var_assignments.get(atom[j+1], atom[j+1])
                     for j in range(len(atom)-1)])
                # assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                if ((self.exclude_symmetric and ground_atom[1] == ground_atom[2]) or
                    (self.exclude_query and ground_atom == q)):
                    is_good = False
                    break
                body_atoms.append(ground_atom)

            if is_good:
                new_groundings.add(((q,), tuple(body_atoms)))
                added += 1
                if self.limit is not None and added >= self.limit:
                    break
      self.rule2groundings[rule.name].update(new_groundings)


# Like a DomainFullGrounder but does not ground the query first,
# al; atoms are considered in a flat manner.
class NonHornDomainFullGrounder(Engine):

    def __init__(self, rules: List[Rule],
                 domains:Dict[str, Domain],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 pure_adaptive: bool=False,
                 limit: int=None):
        self.rules = rules
        self._name2rule = {rule.name: rule for rule in rules}
        self.domains = domains
        self.domain2adaptive_constants = domain2adaptive_constants
        self.pure_adaptive = pure_adaptive
        if self.pure_adaptive:
            assert self.domain2adaptive_constants, (
                'Need adaptive variable if in pure adaptive mode.')
        self.limit = limit

    def _init_internals(self):
        self.rule2groundings = {}
        for rule in self.rules:
            #if rule.name not in self.rule2groundings:
            self.rule2groundings[rule.name] = set()

    #@lru_cache
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []

        self._init_internals()
        for rule in self.rules:
            self.ground_one_rule(rule, queries)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name:
                   RuleGroundings(rule_name,
                                  sorted(list(groundings),
                                         key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name:
                   RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}

        print('GROUNDING DONE', sum([len(r.groundings) for r in ret]))
        return ret

    def ground_one_rule(self, rule: Rule, queries: List[Tuple]) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
      # We have a rule like A(x,y) ^ B(y,z) => C(x,z), D(...)
      rule_atoms = rule.head + rule.body

      new_groundings = set()

      added = 0
      for i,q in enumerate(queries):
        accept = False
        for j,rule_atom in enumerate(rule_atoms):
            if q[0] != rule_atom[0]:  # atom does not match.
                continue
            # Get the variable assignments from the head.
            # If q = Married(Marco,Alice) and head = Married(x,y) then
            # var_assignments = {x: Marco, y=Alice}
            query_var_assignments = {v: a for v, a in zip(rule_atom[1:], q[1:])}

            # Iterate over variables that are not ground by the query.
            vars_to_ground = {v:d for v,d in rule.vars2domain.items()
                              if v not in query_var_assignments}

            if self.pure_adaptive:
                ground_var_groups = [self.domain2adaptive_constants.get(d, [])
                                     for d in vars_to_ground.values()]
            elif self.domain2adaptive_constants is not None:
                ground_var_groups = [self.domains[d].constants +
                                     self.domain2adaptive_constants.get(d, [])
                                     for d in vars_to_ground.values()]
            else:
                ground_var_groups = [self.domains[d].constants
                                     for d in vars_to_ground.values()]

            for ground_vars in product(*ground_var_groups):
                var_assignments = {k:v for k,v in zip(
                    vars_to_ground.keys(), ground_vars)}
                var_assignments.update(query_var_assignments)

                atoms = []
                for atom in rule_atoms:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    atoms.append(ground_atom)
                new_groundings.add(((q,), tuple(atoms)))
                added += 1

                if self.limit is not None and added >= self.limit:
                    break
      self.rule2groundings[rule.name].update(new_groundings)

#! /bin/python3
#from memory_profiler import profile
import copy
from itertools import product
import time
from typing import List, Set, Tuple, Dict, Union

from ns_lib.logic.commons import Atom, Domain, Rule, RuleGroundings
from ns_lib.grounding.engine import Engine


class AtomIndex():
    def __init__(self, facts: List[Tuple[str, str, str]]):
        _index = {}
        for f in facts:
            r_idx, s_idx, o_idx = f
            key = (r_idx, None, None)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = (r_idx, s_idx, None)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = (r_idx,  None, o_idx)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = f
            _index[key] = set([f])

        # Store tuple instead of sets.
        self._index = {k: tuple(v) for k, v in _index.items()}

    def get_matching_atoms(self,
                           atom: Tuple[str, str, str]) -> Tuple[Tuple[str, str, str]]:
        return self._index.get(atom, [])

def get_atoms(r2g:Dict[str, Set[Tuple[Tuple, Tuple]]]) -> Set[Tuple]:
    atoms = set()
    for r, g in r2g.items():
        for rule_atoms in g:
            for atom in rule_atoms[0]:
                atoms.add(atom)
            for atom in rule_atoms[1]:
                atoms.add(atom)
    return atoms

def get_atoms_on_groundings(groundings:Set[Tuple[Tuple, Tuple]]) -> Set[Tuple]:
    atoms = set()
    for rule_atoms in groundings:
        for atom in rule_atoms[0]:  # head
            atoms.add(atom)
        for atom in rule_atoms[1]:  # tail
            atoms.add(atom)
    return atoms

# res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
#@profile
def approximate_backward_chaining_grounding_one_rule(
    domains: Dict[str, Domain],
    domain2adaptive_constants: Dict[str, List[str]],
    pure_adaptive: bool,
    rule: Rule,
    queries: List[Tuple],
    fact_index: AtomIndex,
    # Max number of unknown facts to expand:
    # 0 implements the known_body_grounder
    # -1 or body_size implements pure unrestricted backward_chaining (
    max_unknown_fact_count: int,
    res: Set[Tuple[Tuple, Tuple]]=None,
    # proof is a dictionary:
    # atom -> list of atoms needed to be proved in the application of the rule.
    proofs: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Tuple]]]=None) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
    #start = time.time()
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]
    build_proofs: bool = (proofs is not None)

    new_ground_atoms = set()

    for q in queries:
      #print('Q', q)
      if q[0] != head[0]:  # predicates must match.
        continue

      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}

      for i in range(len(rule.body)):
        # Ground atom by replacing variables with constants.
        # The result is the partially ground atom A('Antonio',None)
        # with None indicating unground variables.
        body_atom = rule.body[i]
        print('BODY_ATOM', body_atom)
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_ground_vars.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        if all(ground_body_atom[1:]):
            groundings = (ground_body_atom,)
        else:
            # Tuple of atoms matching A(Antonio,None) in the facts.
            # This is the list of ground atoms for the i-th atom in the body.
            # groundings = fact_index.get_matching_atoms(ground_body_atom)
            groundings = fact_index._index.get(ground_body_atom, [])

        if len(rule.body) == 1:
            # Shortcut, we are done, the clause has no free variables.
            # Return the groundings.
            new_ground_atoms.add(((q,), groundings))
            continue

        for ground_atom in groundings:
            #print('GROUND ATOM', ground_atom)
            # This loop is only needed to ground at least one atom in the body
            # of the formula. Otherwise it would be enough to start with the
            # loop for ground_vars in product(...) but it would often expand
            # into many nodes. The current solution does not allow to fully
            # recover a classical backward_chaining, though.
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})

            free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                               if v not in head_body_ground_vars]
            free_vars = [vd[0] for vd in free_var2domain]
            if pure_adaptive:
                ground_var_groups = [domain2adaptive_constants.get(vd[1], [])
                                     for vd in free_var2domain]
            elif domain2adaptive_constants is not None:
                ground_var_groups = [domains[vd[1]].constants +
                                     domain2adaptive_constants.get(vd[1], [])
                                     for vd in free_var2domain]
            else:
                ground_var_groups = [domains[vd[1]].constants
                                     for vd in free_var2domain]

            # Iterate over the groundings of the free vars.
            # If no free vars are available, product returns a single empty
            # tuple, meaning that we still correctly enter in the following
            # for loop for a single round.
            # print('FREE VARS_SPAN', list(product(*ground_var_groups)))
            for ground_vars in product(*ground_var_groups):
                var2ground = dict(zip(free_vars, ground_vars))
                full_ground_vars = {**head_body_ground_vars, **var2ground}
                #print('FULL_VARS', full_ground_vars)

                accepted: bool = True
                body_grounding = []
                if build_proofs:
                    body_grounding_to_prove = []
                unknown_fact_count: int = 0
                for j in range(len(rule.body)):
                    if i == j:
                        new_ground_atom = ground_atom
                        # by definition as it is coming from the groundings.
                        is_known_fact = True
                    else:
                        body_atom2 = rule.body[j]
                        new_ground_atom = (body_atom2[0], ) + tuple(
                            [full_ground_vars.get(body_atom2[k+1], None)
                             for k in range(len(body_atom2)-1)])
                        is_known_fact = (fact_index._index.get(
                            new_ground_atom, None) is not None)
                    #print('NEW GROUND ATOM', new_ground_atom, is_known_fact)

                    assert all(new_ground_atom), (
                        'Unexpected free variables in %s' %
                        str(new_ground_atom))

                    if not is_known_fact and (
                            max_unknown_fact_count < 0 or
                            unknown_fact_count < max_unknown_fact_count):
                        body_grounding.append(new_ground_atom)
                        if build_proofs:
                            body_grounding_to_prove.append(new_ground_atom)
                        unknown_fact_count += 1
                        #print('ACCEPTED UNKNOWN', new_ground_atom, unknown_fact_count)
                    elif is_known_fact:
                        body_grounding.append(new_ground_atom)
                        #print('ACCEPTED KNOWN')
                    else:
                        #print('DISCARD', unknown_fact_count, max_unknown_fact_count)
                        accepted = False
                        break
                #print('BODY GROUNDING', body_grounding)

                if accepted:
                    #print('ADDED', q, '->', tuple(body_grounding), 'TO_PROVE',          str(body_grounding_to_prove) if build_proofs else '')
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
                    if build_proofs:
                        proofs.append((q, body_grounding_to_prove))

    #end = time.time()
    #print('NUM_GROUNDINGS', len(new_ground_atoms), 'TIME', end - start)
    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)

# res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
#@profile
def backward_chaining_grounding_one_rule(
    domains: Dict[str, Domain],
    domain2adaptive_constants: Dict[str, List[str]],
    pure_adaptive: bool,
    rule: Rule,
    queries: List[Tuple],
    fact_index: AtomIndex,
    res: Set[Tuple[Tuple, Tuple]]=None) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]

    new_ground_atoms = set()

    for q in queries:
      if q[0] != head[0]:  # predicates must match.
        continue

      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}

      free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                         if v not in head_ground_vars]
      free_vars = [vd[0] for vd in free_var2domain]
      if pure_adaptive:
          ground_var_groups = [domain2adaptive_constants.get(vd[1], [])
                               for vd in free_var2domain]
      elif domain2adaptive_constants is not None:
          ground_var_groups = [domains[vd[1]].constants +
                               domain2adaptive_constants.get(vd[1], [])
                               for vd in free_var2domain]
      else:
          ground_var_groups = [domains[vd[1]].constants
                               for vd in free_var2domain]

      # Iterate over the groundings of the free vars.
      # If no free vars are available, product returns a single empty
      # tuple, meaning that we still correctly enter in the following
      # for loop for a single round.
      #print('FREE VARS_SPAN', list(product(*ground_var_groups)))
      for ground_vars in product(*ground_var_groups):
          var2ground = dict(zip(free_vars, ground_vars))
          full_ground_vars = {**head_ground_vars, **var2ground}
          body_grounding = []
          for j in range(len(rule.body)):
              body_atom = rule.body[j]
              new_ground_atom = (body_atom[0], ) + tuple(
                  [full_ground_vars.get(body_atom[k+1], None)
                   for k in range(len(body_atom)-1)])
              body_grounding.append(new_ground_atom)
          new_ground_atoms.add(((q,), tuple(body_grounding)))

    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)


#@profile
def PruneIncompleteProofs(rule2groundings: Dict[str, Set[Tuple[Tuple, Tuple]]],
                          rule2proofs:Dict[str, List[Tuple[Tuple, List[Tuple]]]],
                          fact_index: AtomIndex,
                          num_steps: int) ->  Dict[str, Set[Tuple[Tuple, Tuple]]]:
    #for rn,g in rule2groundings.items():
    #    print('RIN', rn, len(g))
    atom2proved: Dist[Tuple[str, str, str], bool] = {}

    # This loop iteratively finds the atoms that are already proved.
    for i in range(num_steps):
        for rule_name,proofs in rule2proofs.items():
            for query_and_proof in proofs:
                query, proof = query_and_proof[0], query_and_proof[1]
                if query not in atom2proved or not atom2proved[query]:
                    atom2proved[query] = all(
                        [atom2proved.get(a, False)
                         # This next check is useless as atoms added in the proofs
                         # are by definition not proved already in the data.
                         # or fact_index._index.get(a, None) is not None)
                         for a in proof])

    # Now atom2proved has all proved atoms. Scan the groundings and keep only
    # the ones that have been proved within num_steps:
    pruned_rule2groundings = {}
    for rule_name,groundings in rule2groundings.items():
        pruned_groundings = []
        for g in groundings:
            head_atoms = g[0]
            # all elements in the grounding are either in the training data
            # or they are provable using the rules,
            if all([(atom2proved.get(a, False) or
                     fact_index._index.get(a, None) is not None)
                    for a in head_atoms]):
                pruned_groundings.append(g)
        pruned_rule2groundings[rule_name] = set(pruned_groundings)
    #for rn,g in pruned_rule2groundings.items():
    #    print('ROUT', rn, len(g))
    return pruned_rule2groundings


class ApproximateBackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 pure_adaptive: bool=False,
                 max_unknown_fact_count: int=1,
                 max_unknown_fact_count_last_step: int=1,
                 num_steps: int=1,
                 max_groundings_per_rule: int=-1,  # to speedup the computation.
                 # Whether the groundings should be accumulated across calls.
                 accumulate_groundings: bool=False,
                 prune_incomplete_proofs: bool=True):
        self.max_unknown_fact_count = max_unknown_fact_count
        self.max_unknown_fact_count_last_step = max_unknown_fact_count_last_step
        self.num_steps = num_steps
        self.accumulate_groundings = accumulate_groundings
        self.prune_incomplete_proofs = prune_incomplete_proofs
        self.max_groundings_per_rule = max_groundings_per_rule
        self.rules = rules
        self.domains = domains
        self.domain2adaptive_constants = domain2adaptive_constants
        self.pure_adaptive = pure_adaptive
        self.facts = [a if isinstance(a,Tuple) else a.toTuple()
                      if isinstance(a,Atom) else Atom(s=a).toTuple()
                      for a in facts]
        # self.facts = facts
        for rule in self.rules:
            assert len(rule.head) == 1, (
                '%s is not a Horn clause' % str(rule))
        self._fact_index = AtomIndex(self.facts)
        self.relation2queries = {}
        self.rule2groundings = {}
        self.rule2proofs = {}

    def _init_internals(self, queries: List[Tuple], clean: bool):
        self.relation2queries = {}  # reset
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = set()
            self.relation2queries[q[0]].add(q)

        # If clean=False, groundings are incrementally added.
        for rule in self.rules:
            if clean or rule.name not in self.rule2groundings:
                self.rule2groundings[rule.name] = set()
                self.rule2proofs[rule.name] = []

    # Ground a batch of queries, the result is cached for speed.
    #@profile
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []

        # When accumulating groundings, we keep a single large set of
        # groundings that are reused over all batches.
        self._init_internals(queries, clean=(not self.accumulate_groundings))
        # Keeps track of the queris already processed for this rule.
        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        for step in range(self.num_steps):
            # print('STEP', step)
            for rule in self.rules:
                # Here we assume to have a Horn clause, fix it.
                queries_per_rule = list(
                    self.relation2queries.get(rule.head[0][0], set()))
                if not queries_per_rule:
                    continue
                approximate_backward_chaining_grounding_one_rule(
                    self.domains,
                    self.domain2adaptive_constants,
                    self.pure_adaptive,
                    rule, queries_per_rule, self._fact_index,
                    # max_unknown_fact_count
                    (self.max_unknown_fact_count if step < self.num_steps - 1
                     else self.max_unknown_fact_count_last_step),
                    # Output added here.
                    res=self.rule2groundings[rule.name],
                    # Proofs added here.
                    proofs=(self.rule2proofs[rule.name]
                            if self.prune_incomplete_proofs else None))
                # Update the list of processed rules.
                self._rule2processed_queries[rule.name].update(queries_per_rule)
                # print(step, 'PROCESSED', len(queries_per_rule), len(self._rule2processed_queries[rule.name]))

            if step == self.num_steps - 1:
                break

            # Get the queries for the next iteration.
            new_queries = set()
            for rule in self.rules:
                groundings = self.rule2groundings[rule.name]
                # Get the new queries left to prove, these facts that are not
                # been processed already and that are not known facts.
                new_queries.update(
                    [a for a in get_atoms_on_groundings(groundings)
                     if a not in self._rule2processed_queries[rule.name] and
                     self._fact_index._index.get(a, None) is None])
            # print(step, 'NEW Q', list(new_queries)[:10], 'FROM', len(groundings))
            self._init_internals(list(new_queries), clean=False)

        if self.prune_incomplete_proofs:
            self.rule2groundings = PruneIncompleteProofs(self.rule2groundings,
                                                         self.rule2proofs,
                                                         self._fact_index,
                                                         self.num_steps)

        # This should be done after sorting the groundings to ensure the output
        # to be deterministic.
        if self.max_groundings_per_rule > 0:
            self.rule2groundings = {rule_name:set(list(groundings)[:self.max_groundings_per_rule])
                                    for rule_name,groundings in self.rule2groundings.items()}

        #print('R', self.rule2groundings)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name: RuleGroundings(
                rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name: RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}

        return ret


class BackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule],
                 facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 pure_adaptive: bool=False,
                 num_steps: int=1,
                 # Whether the groundings should be accumulated across calls.
                 accumulate_groundings: bool=False):
        self.num_steps = num_steps
        self.accumulate_groundings = accumulate_groundings
        self.rules = rules
        self.domains = domains
        self.domain2adaptive_constants = domain2adaptive_constants
        self.pure_adaptive = pure_adaptive
        self.facts = [a if isinstance(a,Tuple) else a.toTuple()
                      if isinstance(a,Atom) else Atom(s=a).toTuple()
                      for a in facts]
        # self.facts = facts
        for rule in self.rules:
            assert len(rule.head) == 1, (
                '%s is not a Horn clause' % str(rule))
        self._fact_index = AtomIndex(self.facts)
        self.relation2queries = {}
        self.rule2groundings = {}
        self.rule2proofs = {}

    def _init_internals(self, queries: List[Tuple], clean: bool):
        self.relation2queries = {}  # reset
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = set()
            self.relation2queries[q[0]].add(q)

        # If clean=False, groundings are incrementally added.
        for rule in self.rules:
            if clean or rule.name not in self.rule2groundings:
                self.rule2groundings[rule.name] = set()
                self.rule2proofs[rule.name] = []

    # Ground a batch of queries, the result is cached for speed.
    #@profile
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []

        # When accumulating groundings, we keep a single large set of
        # groundings that are reused over all batches.
        self._init_internals(queries, clean=(not self.accumulate_groundings))
        # Keeps track of the queris already processed for this rule.
        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        for step in range(self.num_steps):
            # print('STEP', step)
            for rule in self.rules:
                # Here we assume to have a Horn clause, fix it.
                queries_per_rule = list(
                    self.relation2queries.get(rule.head[0][0], set()))
                if not queries_per_rule:
                    continue
                backward_chaining_grounding_one_rule(
                    self.domains,
                    self.domain2adaptive_constants,
                    self.pure_adaptive,
                    rule, queries_per_rule, self._fact_index,
                    # Output added here.
                    res=self.rule2groundings[rule.name])
                # Update the list of processed rules.
                self._rule2processed_queries[rule.name].update(queries_per_rule)
                # print(step, 'PROCESSED', len(queries_per_rule), len(self._rule2processed_queries[rule.name]))

            if step == self.num_steps - 1:
                break

            # Get the queries for the next iteration.
            new_queries = set()
            for rule in self.rules:
                groundings = self.rule2groundings[rule.name]
                # Get the new queries left to prove, these facts that are not
                # been processed already and that are not known facts.
                new_queries.update(
                    [a for a in get_atoms_on_groundings(groundings)
                     if a not in self._rule2processed_queries[rule.name] and
                     self._fact_index._index.get(a, None) is None])
            # print(step, 'NEW Q', list(new_queries)[:10], 'FROM', len(groundings))
            self._init_internals(list(new_queries), clean=False)

        #print('R', self.rule2groundings)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name: RuleGroundings(
                rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name: RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}

        return ret

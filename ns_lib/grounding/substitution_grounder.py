import abc
from typing import Dict, List, Tuple
from ns_lib.logic.commons import Rule, RuleGroundings
from ns_lib.grounding.engine import Engine

#############################################
class SubstitutionGrounder(Engine):

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    @abc.abstractmethod
    def queries_to_substitutions(self, queries, rule):
        pass

    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:
        res = {}
        for rule in self.rules:
            groundings = []
            # for query in queries:  # avoid 1 call per query.
            for substitution in self.queries_to_substitutions(queries, rule):
                body_atoms = []
                for atom in rule.body:
                    ground_atom = (atom[0], ) + tuple(
                        [substitution.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    body_atoms.append(ground_atom)

                head_atoms = []
                for atom in rule.head:
                    ground_atom = (atom[0], ) + tuple(
                        [substitution.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    head_atoms.append(ground_atom)
                groundings.append((tuple(head_atoms), tuple(body_atoms)))
                print('G', groundings[-1])

            res[rule.name] = RuleGroundings(rule.name, groundings=groundings)
        return res

#############################################
class FlatGrounder(SubstitutionGrounder):

    def __init__(self, rules: List[Rule]):
        super().__init__(rules)

    def queries_to_substitutions(self, queries, rule):
        substitutions = []
        for query in queries:
            assert len(rule.vars) == len(query[1:])
            for atom in rule.head + rule.body:
                if query[0] != atom[0]:
                    continue

                substitutions.append(
                    {variable:constant
                     for variable,constant in zip(atom[1:],query[1:])})
                break
        return substitutions

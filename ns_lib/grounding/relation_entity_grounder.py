import random
from typing import Dict, List, Tuple
from ns_lib.grounding.substitution_grounder import SubstitutionGrounder
from ns_lib.logic.commons import Rule
from statistics import mean, median


###############################################
class RelationEntityGraph(object):
    def __init__(self, facts: List[Tuple]):
        # entity -> neighbour_entity1, neighbour_entity2, ...]
        self.adj_graph: Dict[str, List[str]] = {}
        self.adj_graph_by_relation: Dict[str, Dict[str, List[str]]] = {}
        # entity -> relation1, relation2, ...]
        self.adj_graph_edges: Dict[str, List[str]] = {}
        for f in facts:
            if f[1] not in self.adj_graph:
                self.adj_graph[f[1]] = []
                self.adj_graph_by_relation[f[1]] = {}
                self.adj_graph_edges[f[1]] = []
            self.adj_graph[f[1]].append(f[2])
            self.adj_graph_edges[f[1]].append(f[0])
            if f[0] not in self.adj_graph_by_relation[f[1]]:
                self.adj_graph_by_relation[f[1]][f[0]] = []
            self.adj_graph_by_relation[f[1]][f[0]].append(f[2])
            if f[2] not in self.adj_graph:
                self.adj_graph[f[2]] = []
                self.adj_graph_by_relation[f[2]] = {}
                self.adj_graph_edges[f[2]] = []
            self.adj_graph[f[2]].append(f[1])
            self.adj_graph_edges[f[2]].append(f[0])
            if f[0] not in self.adj_graph_by_relation[f[2]]:
                self.adj_graph_by_relation[f[2]][f[0]] = []
            self.adj_graph_by_relation[f[2]][f[0]].append(f[1])

        print('RelationEntityGraph', self.stats())

    def get_neighbors(self, entity: str) -> List[str]:
        if entity not in self.adj_graph:
            return []
        return self.adj_graph[entity]

    def get_edges(self, entity: str) -> List[str]:
        if entity not  in self.adj_graph_edges:
            return []
        return self.adj_graph_edges[entity]

    def get_neighbors_for_relation(self, entity: str,
                                   relation: str) -> List[str]:
        r2l = self.adj_graph_by_relation.get(entity, None)
        if r2l is None or relation not in r2l:
            return []
        return r2l[relation]

    def stats(self) -> Dict[str, float]:
        values = self.adj_graph.values()
        min_list_length = min([len(l) for l in values])
        max_list_length = max([len(l) for l in values])
        mean_list_length = mean([len(l) for l in values])
        median_list_length = median([len(l) for l in values])
        return {'min': min_list_length,
                'max': max_list_length,
                'mean': mean_list_length,
                'median': median_list_length}

###################################################
#TODO: add domain support.
class RelationEntityGraphGrounder(SubstitutionGrounder):

    def __init__(self, rules: List[Rule], facts: List[Tuple],
                 build_cartesian_product: bool=True,
                 max_elements: int=None):
        self.rules = rules
        self.max_elements = max_elements
        self.relation_entity_graph = RelationEntityGraph(facts)
        print('Relation Entity Stats', self.relation_entity_graph.stats())
        if self.max_elements is not None:
            self.max_elements_per_arm = self.max_elements // 2

    def queries_to_substitutions(self, queries, rule):
        assert len(rule.vars) <= 3

        substitutions = []
        rule_atoms = rule.body + rule.head
        for query in queries:
            # Take all the atoms in the rule having the the query relation.
            # Example A(x,y) ^ A(y,z) => C(x,z),  Qury A(c1,c2)
            # query_relation_atoms = {A(x, y), A(y,z)}
            query_relation_atoms = [a for a in rule_atoms if query[0] == a[0]]
            if len(query_relation_atoms) == 0:
                continue

            # Take all the atoms in the rule not having the the query relation.
            # non_query_atoms = {B(y,z), C(x,z)}
            non_query_relation_atoms = [
                a for a in rule_atoms if query[0] != a[0]]

            # Short cut for speed, this is the pointwise case with no
            # free variables after grounding the query.
            if len(rule.vars) == 2:  # only 2 vars grounded by the query
                for a in query_relation_atoms:
                    substitutions.append(
                        {v: q for v,q in zip(a[1:],query[1:])}) # {X:a, Y:b}
                continue

            # Get neighbours for (c1,c2) and merge them.
            neighbours1 = self.relation_entity_graph.get_neighbors(query[1])
            neighbours2 = self.relation_entity_graph.get_neighbors(query[2])
            if self.max_elements_per_arm is not None:
                if len(neighbours1) > self.max_elements_per_arm:
                  neighbours1 = random.sample(neighbours1,
                                              self.max_elements_per_arm)
                if len(neighbours2) > self.max_elements_per_arm:
                  neighbours2 = random.sample(neighbours2,
                                              self.max_elements_per_arm)
            neighbours = neighbours1 + neighbours2
            if len(neighbours) == 0:
                continue
            neighbours = sorted(list(set(neighbours)))

            for a in query_relation_atoms:
                # Select one query atom, like for example A(y,z),
                # align with the query to get the assignment {y:c1, z:c2}
                var_assignments = {v:q for v,q in zip(a[1:], query[1:])}
                # Get the free vars, [x] in the example
                free_vars = [v for v in rule.vars if v not in var_assignments]
                assert len(free_vars) == 1, 'Free Vars:%s over rule_vars:%s' % (
                    str(free_vars), str(rule.vars))
                # Replace the free variable with the constant candidates.
                for n in neighbours:
                    var_assignments[free_vars[0]] = n
                    substitutions.append(
                        {**var_assignments, **{free_vars[0]: n}})
        return substitutions

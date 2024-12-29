from typing import Dict, List, Tuple
from ns_lib.logic.commons import Predicate, Domain
from ns_lib.grounding.engine import Engine
from collections import defaultdict
import tensorflow as tf




class RuleTensorGroundings():

    def __init__(self, name: str, groundings: Tuple[tf.Tensor, tf.Tensor]):

        self.name = name
        self.groundings = groundings

#############################################
class TensorFlatGrounder(Engine):

    def __init__(self, rules):
        self.rules = rules
        rule = rules[0]
        self.rule_name = rule.name
        self.num_head_rel = len(rule.head)
        self.head_relations = tf.expand_dims(tf.expand_dims(tf.constant([a[0] for a  in rule.head]), 0),-1) # [1, num_relations, 1]

        self.num_body_rel = len(rule.body)
        self.body_relations = tf.expand_dims(tf.expand_dims(tf.constant([a[0] for a  in rule.body]), 0), -1)  # [1, num_relations, 1]

    @tf.function
    def ground(self, facts: tf.Tensor, queries: tf.Tensor,**kwargs) -> Dict[str, Tuple[tf.Tensor, tf.Tensor]]:

        queries = tf.expand_dims(queries, 1) # [batch, 1, 3]
        substitutions = queries[:,:, 1:]

        substitutions_head = tf.tile(substitutions, [1, self.num_head_rel, 1])
        head_relations = tf.tile(self.head_relations, [tf.shape(queries)[0], 1, 1])
        groundings_head = tf.concat((head_relations, substitutions_head), axis=-1)

        substitutions_body = tf.tile(substitutions, [1, self.num_body_rel, 1])
        body_relations = tf.tile(self.body_relations, [tf.shape(queries)[0], 1, 1])
        groundings_body = tf.concat((body_relations, substitutions_body), axis=-1)

        rule = {self.rule_name: (groundings_body,groundings_head)}

        return  rule






class TensorLogicSerializer():

    def __init__(self, predicates: List[Predicate], domains: List[Domain]):
        self.predicates = predicates
        self.domains = domains

        self.constant_to_global_index = defaultdict()
        for domain in domains:
            self.constant_to_global_index[domain.name]  = \
                tf.lookup.StaticHashTable(
                     tf.lookup.KeyValueTensorInitializer(tf.constant(domain.constants),
                                                         tf.range(len(domain.constants))),
                    default_value=-1)

        self.predicate_to_domains = {}
        for predicate in predicates:
            self.predicate_to_domains[predicate.name] = [domain.name for domain in predicate.domains]


        self.predicates_table = tf.lookup.StaticHashTable(
                     tf.lookup.KeyValueTensorInitializer(tf.constant([p.name for p in self.predicates]),
                                                         tf.range(len(self.predicates))),
                    default_value=-1)

        self.domain_to_local_constant_index = {}
        for domain in self.domains:
            self.domain_to_local_constant_index[domain.name] = tf.lookup.experimental.DenseHashTable(
                                                                    tf.string,
                                                                    tf.int32,
                                                                    -1,
                                                                    "<<EMPTY>>",
                                                                    "<<DEL>>")


    # @tf.function
    def serialize(self, queries: tf.Tensor, rule_groundings: Dict[str, Tuple[tf.Tensor, tf.Tensor]]):

        # We store all the atom strings used in either the queries or the groundings
        all_atoms_strings_with_duplicates = []

        # Create string representation for each atom in the queries
        queries_strings = tf.strings.reduce_join(queries, axis=-1,separator="<>")
        all_atoms_strings_with_duplicates.append(tf.reshape(queries_strings, [-1]))

        # Create string representation for each atom in the groundings
        rule_groundings_strings = {}
        for name, groundings in rule_groundings.items():
            body_groundings = groundings[0]
            head_groundings = groundings[1]
            body_groundings_strings = tf.strings.reduce_join(body_groundings, axis=-1, separator="<>")
            head_groundings_strings = tf.strings.reduce_join(head_groundings, axis=-1, separator="<>")
            rule_groundings_strings[name] = (body_groundings_strings, head_groundings_strings)
            all_atoms_strings_with_duplicates.append(tf.reshape(body_groundings_strings, [-1]))
            all_atoms_strings_with_duplicates.append(tf.reshape(head_groundings_strings, [-1]))


        all_atoms_strings_with_duplicates = tf.concat(all_atoms_strings_with_duplicates, axis=0)

        # Create a tensor of unique strings of atoms
        all_atoms_strings, _  = tf.unique(all_atoms_strings_with_duplicates)
        all_atoms = tf.strings.split(all_atoms_strings, sep="<>").to_tensor()


        # Sorting them by bucketing  per predicate
        relations_in_atoms  = all_atoms[:,0]
        relations_in_atoms = self.predicates_table.lookup(relations_in_atoms)

        all_atoms_per_predicate = {}
        for i,r in enumerate(self.predicates):
            k = tf.squeeze(tf.where(relations_in_atoms == i))
            all_atoms_per_predicate[r.name] = tf.gather(params=all_atoms, indices=k)



        constants_in_batch_per_domain = defaultdict(list)  # a dict mapping each domain to the list of constant strings used in the atoms

        all_atoms_sorted_by_predicate = [] # equivalent to all_atoms, but now the atoms are sorted per predicate (flat version of all_atoms_per_predicate)
        for predicate in self.predicates:
            atoms = all_atoms_per_predicate[predicate.name]
            all_atoms_sorted_by_predicate.append(atoms)

            for i, domain in enumerate(predicate.domains):
                constants_in_batch_per_domain[domain.name].append(atoms[:,i+1]) # +1 because we need to skip the relation in position 0
        all_atoms_sorted_by_predicate = tf.concat(all_atoms_sorted_by_predicate, axis=0)


        domain_to_global = {}  # The actual output X_domains
        for domain in self.domains:
            constants_in_batch_per_domain[domain.name] = tf.concat(constants_in_batch_per_domain[domain.name], axis=0)
            constants_in_batch_per_domain[domain.name] = tf.unique(constants_in_batch_per_domain[domain.name])[0]
            table = self.domain_to_local_constant_index[domain.name]
            table.insert(keys= constants_in_batch_per_domain[domain.name], values = tf.range(len(constants_in_batch_per_domain[domain.name])))
            domain_to_global[domain.name] = self.constant_to_global_index[domain.name].lookup(constants_in_batch_per_domain[domain.name])



        predicate_to_constant_tuples = defaultdict() # A_predicates
        for predicate in self.predicates:
            atoms = all_atoms_per_predicate[predicate.name]
            predicate_to_constant_tuples[predicate.name] = tf.stack([self.domain_to_local_constant_index[domain.name].lookup(atoms[:,i+1]) for i, domain in enumerate(predicate.domains)], axis=1)


        # We recreate the strings of the unique atoms grouped and sorted by predicate
        all_atoms_sorted_by_predicate =  tf.strings.reduce_join(all_atoms_sorted_by_predicate, axis=-1, separator="<>")
        all_atoms_table = tf.lookup.StaticHashTable(
                     tf.lookup.KeyValueTensorInitializer(all_atoms_sorted_by_predicate,
                                                         tf.range(len(all_atoms_sorted_by_predicate))),
                    default_value=-1)


        # Query indexing
        queries = all_atoms_table.lookup(queries_strings)


        # Rule Groundings indexing
        rule_groundings_indices = {}
        for name, (body_groundings_strings, head_groundings_strings) in  rule_groundings_strings.items():

            rule_groundings_indices[name] = (all_atoms_table.lookup(body_groundings_strings),
                                             all_atoms_table.lookup(head_groundings_strings))


        for domain in self.domains:
            self.domain_to_local_constant_index[domain.name].remove(constants_in_batch_per_domain[domain.name])


        return domain_to_global, predicate_to_constant_tuples, rule_groundings_indices, queries


if __name__ == '__main__':

    from ns_lib.logic.commons import Predicate, Domain, Rule


    rules = [Rule("rule0", body_atoms=[(r, "X", "Y")
                                            for r in ["r3"]],
                          head_atoms=[(r, "X", "Y") for r in ["r1", "r2"]])]
    grounder  = TensorFlatGrounder(rules)

    queries = [("r1", "a", "b"), ("r2", "d", "f")]
    queries = tf.constant(queries)

    groundings = grounder.ground(facts = None, queries = queries)

    d = Domain("domain", constants=["a", "b", "d", "f"])
    r1 = Predicate("r1", domains=[d,d])
    r2 = Predicate("r2", domains=[d,d])
    r3 = Predicate("r3", domains=[d,d])
    serializer = TensorLogicSerializer(predicates=[r1,r2,r3], domains=[d])

    print(serializer.serialize(queries,groundings))
    queries = tf.constant([("r1", "a", "b"), ("r2", "d", "f"), ("r3", "d", "a")])
    print(serializer.serialize(queries,groundings))
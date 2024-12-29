import argparse
from typing import List, Dict
from ns_lib.logic.commons import FOL
from ns_lib.grounding import *

def get_arg(args, name: str, default=None, assert_defined: bool=False):
    value = getattr(args, name) if hasattr(args, name) else default
    if assert_defined:
        assert value is not None, 'Arg %s is not defined: %s' % (name, str(args))
    return value

def BuildGrounder(args, rules: List[Rule], facts: List[Tuple], fol: FOL,
                  domain2adaptive_constants: Dict[str, List[str]]):
    type = get_arg(args, 'grounding_type', 'BackwardChainingGrounder')
    print('Building Grounder', type, flush=True)

    if type == 'ApproximateBackwardChainingGrounder':
        # Requires Horn Clauses.
        return ApproximateBackwardChainingGrounder(
            rules, facts=facts, domains={d.name:d for d in fol.domains},
            domain2adaptive_constants=domain2adaptive_constants,
            pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
            num_steps=get_arg(args, 'backward_chaining_depth', 1),
            max_unknown_fact_count=get_arg(
                args, 'backward_chaining_max_unknown_fact_count', 1),
            max_groundings_per_rule=get_arg(
                args, 'backward_chaining_max_groundings_per_rule', -1),
            prune_incomplete_proofs=get_arg(
                args, 'backward_chaining_prune_incomplete', True))

    elif type == 'BackwardChainingGrounder':
        # Requires Horn Clauses.
        return BackwardChainingGrounder(
            rules, facts=facts,
            domains={d.name:d for d in fol.domains},
            domain2adaptive_constants=domain2adaptive_constants,
            pure_adaptive=get_arg(args, 'engine_pure_adaptive', False),
            num_steps=get_arg(args, 'backward_chaining_depth', 1))

    elif type == 'RelationEntityGraphGrounder':
        # Requires Horn Clauses.
        return RelationEntityGraphGrounder(
            rules, facts=facts,
            # TODO: Domain support is not added yet.
            #domains={d.name:d for d in fol.domains},
            #domain2adaptive_constants=domain2adaptive_constants,
            build_cartesian_product=True,
            max_elements=get_arg(
                args, 'relation_entity_grounder_max_elements', -1))

    elif type == 'KnownBodyGrounder':
        # Requires Horn Clauses.
        return KnownBodyGrounder(rules, facts=facts)

    elif type == 'DomainFullGrounder':
        return DomainFullGrounder(
            rules, domains={d.name:d for d in fol.domains},
            domain2adaptive_constants=domain2adaptive_constants)

    elif type == 'DomainBodyFullGrounder':
        # It currently requires Horn Clauses, but it could be extended.
        return DomainBodyFullGrounder(
            rules, domains={d.name:d for d in fol.domains},
            domain2adaptive_constants=domain2adaptive_constants,
            pure_adaptive=get_arg(args, 'engine_pure_adaptive', False))

    elif type == 'NonHornDomainFullGrounder':
        # It works with any clause (even non Horn), but it has to be tested.
        return NonHornDomainFullGrounder(
            rules, domains={d.name:d for d in fol.domains},
            domain2adaptive_constants=domain2adaptive_constants,
            pure_adaptive=get_arg(args, 'engine_pure_adaptive', False))

    elif type == 'FlatGrounder':
        # This grounder is very fast but it works only for rules with
        # no free variables after the query is selected. It is typically
        # used for rule like A(x,y) ^ B(x,y) ^ ... ^ Z(x,y).
        # If the rule is not in this form the code will exit with an error.
        return FlatGrounder(rules)

    else:
        assert False, 'Unknown grounder %s' % type

    return None

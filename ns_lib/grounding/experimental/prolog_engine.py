from ns_lib.grounding.engine import Engine
from collections import defaultdict
from problog.program import PrologString, SimpleProgram
from problog.logic import Clause, And, Term, Constant,Var
from problog.engine import DefaultEngine
from problog.formula import LogicFormula
import warnings
import os
from typing import Dict


class PrologRule():

    def __init__(self, definition, hard=False):
        self.definition = definition
        self._hard = hard

    @property
    def hard(self):
        return self._hard



class PrologEngine(Engine):


    def __init__(self, predicates, rules:Dict[str, PrologRule], depth = -1, base_facts = None, cache_activated = True, cache_file = "global_cache.dat"):
        super().__init__(rules)
        self.predicates = predicates
        self.engine = DefaultEngine(label_all=True)
        self._rules_dict = {}
        self.depth = depth
        for k,v in self.rules.items():
            signature = self.create_signature(v.definition)
            self._rules_dict[k] = (len(signature[0]),len(signature[1]))
        self.base_facts = base_facts
        self.cache = {}
        self.cache_activated = cache_activated
        self.cache_file = cache_file
        self.read_cache()

    @property
    def arity_dict(self):
        return self._rules_dict

    def _remove_depth(self, term):
        return Term(term.functor, *term.args[:-1])

    def _add_arg(self, term, new_arg, in_front=False):
        if not in_front:
            return Term(term.functor, *(list(term.args)+[new_arg]), p = term.probability)
        else:
            return Term(term.functor, *([new_arg] + list(term.args)), p = term.probability)

    def _add_depth_to_program(self, program, depth):
        """It adds to each rule a variable Depth, to each fact a final variable _ and to each query a constant 'depth'.
        Rules decrease the depth after each application and fail if the allowed depth is 0.

        DISCLAIMER: This method is experimental and does not handle general prolog programs, but only simple ones


        E.g.
        depth=2

        0.1::edge(a,b,_)
        0.1::edge(b,d,_)
        0.1::edge(b,c,_)

        path(X,Y,Depth) :- Depth>0, edge(X,Z,Depth-1), path(Z,Y,Depth-1)
        path(X,Y,Depth) :- Depth>0, edge(X,Y,Depth-1)

        query(edge(a,b,2))
        query(edge(b,d,2))
        query(edge(b,c,2))
        """

        warnings.warn("Using depth is an experimental feature. It may cause unexpected behaviours")

        new_program = SimpleProgram()
        for i, line in enumerate(program):
            if line.functor == "query":
                term = line.args[0]
                new_query = Term("query", self._add_arg(term, Constant(depth)))
                new_program.add_fact(new_query)
            elif isinstance(line, Clause):
                clause = line
                v = Var("Depth")
                head = self._add_arg(clause.head, v)
                if isinstance(clause.body, And):
                    body = clause.body.to_list()
                elif isinstance(clause.body, Term):
                    body = [clause.body]
                else:
                    body = clause.body
                body = And.from_list(
                    [Term(">", v, Constant(0))] + [self._add_arg(i, Term("-", v, Constant(1))) for i in body])
                new_clause = Clause(head, body)
                new_program.add_clause(new_clause)
            elif isinstance(line, Term):
                new_program.add_fact(self._add_arg(line, Var('_')))
        return new_program


    def write_cache(self,key,value):
        self.cache[key] = value
        with open(self.cache_file, "a") as f:
            f.write(str(key)+"\t"+str(value)+"\n")

    def read_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                for line in f.readlines():
                    ls = line.split("\t")
                    self.cache[eval(ls[0])] = eval(ls[1])






    def ground(self, facts, queries, **kwargs):

        key = (facts, queries)
        if key in self.cache and self.cache_activated:
            return self.cache[key]
        else:

            max_depth = self.depth

            if self.rules is None or len(self.rules) == 0:
                return {}

            formulas = [v.definition for v in self.rules.values()]

            #now we create a ProbLog program
            # formulas become rules of the programs
            rules_str = "\n".join([str(id) + "::" + formula.definition + "." for id, formula in self.rules.items()])

            # facts (train data + evidence facts we don't want to predict) are made probabilistic # TODO do we need to make evidence facts probabilistic?? (this means they will appear in the groundings)
            facts_str = "\n".join(["0.1::{}.".format(fact) for fact in facts])
            if self.base_facts is not None:
                facts_str = facts_str + "\n".join(["0.1::{}.".format(fact) for fact in self.base_facts])

            # queries (atoms of whom we compute a proof) are all the train, valid and test facts
            queries_str = "\n".join(["query({}).".format(fact) for fact in queries])

            # we assemble the model
            program = "\n\n".join([facts_str,rules_str,queries_str])
            #TODO assert that input is in the right format (we can abstract this check at the level of Engine, since all engines should do this check

            result = self._ground_program(program, max_depth)

            self.write_cache(key, result)
            # ground the program
            return result


    def create_signature(self, clause):

        if isinstance(clause,str):
            if "." not in clause:
                clause = clause + "."
            prolog = PrologString(clause)
            for c in prolog:
                clause = c
                break
        if isinstance(clause.body, And):
            body = clause.body.to_list()
        elif isinstance(clause.body, list):
            body = clause.body
        else:
            body = [clause.body]
        head = clause.head
        signature = (tuple([term.functor for term in body if term.functor if term.functor in self.predicates]),
                      tuple([head.functor]))
        return signature



    def _ground_program(self, program,max_depth = -1):

        # create base ProbLog program
        prolog = PrologString(program)

        # if max_depth>0, we refactor the program to prove queries up to length N. This is a syntactic change and does not affect the engine
        if max_depth>0:
            prolog = self._add_depth_to_program(prolog, max_depth)

        # we create an empty problog.LogicFormula (conjunction of ground clauses)
        # the flags allows us to obtain a grounding with no simplifications, which is more adherent to the initial rules
        f = LogicFormula(avoid_name_clash=True, keep_order=True, keep_duplicates=True)

        # we prove all the queries and we add the clauses used to a transformed problog.LogicFormula `ground`
        ground = self.engine.ground_all(prolog, target=f)

        """here we start with logictransformers modeling"""

        # we group the clauses by signature.
        rule_id_to_groundings = defaultdict(list)
        all_clauses = [c for c in ground.enum_clauses()]
        all_clauses = set(all_clauses)
        for clause in all_clauses:
            if isinstance(clause, Clause):
                if isinstance(clause.body,And):
                    body = clause.body.to_list()
                else:
                    body = [clause.body]
                head = clause.head
                id = str(head.probability)
                head.probability = None
                # since the depth was handled syntactically we need to remove depths from the atoms
                if max_depth>0:
                    head = self._remove_depth(head)
                    body = [self._remove_depth(b) for b in body]
                g = ([str(term) for term in body],[str(head)])
                rule_id_to_groundings[id].append(g)


        return dict(rule_id_to_groundings)

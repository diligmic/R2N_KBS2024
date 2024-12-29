from collections import defaultdict
import json
from typing import Dict, List, Tuple

from ns_lib.grounding.engine import Engine
from ns_lib.logic.commons import Atom, Rule, RuleGroundings


# Simple Grounder, loading the grounding list from a json file.
class FileGrounder(Engine):

    def __init__(self, filepath, rules: Dict[str, Rule],
                 format="functional"):

        super().__init__(rules)

        if len(rules) > 0:
            self.filepath = filepath

            with open(filepath) as f:
                self.cache = json.load(f)

            for q, r in  self.cache.items():
                for id,gr in r.items():
                    R = []
                    for head, body in gr:
                        body = [Atom(b) for b in body]
                        head = [Atom(h) for h in head]
                        R.append(Rule(body=body, head=head))
                    self.cache[Atom(q, format=format)][id] = R

    @property
    def arity_dict(self):
        res = {k: (len(v.body), len(v.head)) for k,v in self.rules.items()}
        return res


    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:
        if self.rules is None or len(self.rules) == 0:
            return {}

        del facts # they have been already used in the creation of the groundings
        res = defaultdict(set)

        for query in queries:
            if query not in self.cache:
                continue
            r = self.cache[query]
            for id, gr in r.items():
                res[id] = res[id].union(set(gr))
        return {name:RuleGroundings(name, list(res[name]))
                for name in self.rules.keys()}

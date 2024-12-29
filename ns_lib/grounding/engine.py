import abc
from typing import Dict, List, Tuple
from ns_lib.logic.commons import RuleGroundings


class Engine(object):
    """General interface for grounder engines."""

    @abc.abstractmethod
    def ground(self,
               facts:List[Tuple],
               queries:List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:
        pass

    """Count the groundings returned by ground()."""
    @staticmethod
    def count_groundings(grounding_list:List[RuleGroundings]) -> int:
        return sum([len(g.groundings) for g in grounding_list])

class NullEngine(Engine):
    """Null grounder which does not ground anything."""

    def __init__(self):
        self.rules = []

    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:
        return {}

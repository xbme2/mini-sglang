from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .radix_cache import TreeNode



class EvictPolicy(ABC):
    @abstractmethod
    def get_priority(self, node: TreeNode):
        """Get the priority of a node for eviction. """
        pass

class LRUEvictPolicy(EvictPolicy):
    def get_priority(self, node: TreeNode):
        return node.last_access_time

class FIFOEvictPolicy(EvictPolicy):
    def get_priority(self, node: TreeNode):
        return node.creation_time

class LFUEvictPolicy(EvictPolicy):
    def get_priority(self, node: TreeNode):
        return (node.hit_count, node.last_access_time)  


EVICTPOLICY_REGISTRY = {
    "lru": LRUEvictPolicy,
    "fifo": FIFOEvictPolicy,
    "lfu": LFUEvictPolicy,
}
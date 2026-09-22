from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class TreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, TreeNode] = {}
        self._parent: TreeNode | None = None
        self.ref_count: int = 0
        self.hit_count: int = 0
        self.uuid = TreeNode.counter
        TreeNode.counter += 1

        self.last_access_time = tic or time.monotonic()
        self.creation_time = tic or time.monotonic()

        # self.timestamp = tic or time.monotonic()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: TreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> TreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> TreeNode:
        assert 0 < pos < self.length
        parent = self.parent
        # [todo]
        new_node = TreeNode(self.key_fn)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        new_node.hit_count = self.hit_count

        new_node.last_access_time = time.monotonic()
        new_node.creation_time = self.creation_time

        new_node.children[self.key_fn(self._key[pos:])] = self

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: TreeNode) -> bool:
        return self.last_access_time < other.last_access_time


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: TreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device, evict_policy: str = "lru"):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = TreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

        from .evict_policy import EVICTPOLICY_REGISTRY

        self.evict_strategy = EVICTPOLICY_REGISTRY.get(evict_policy)()

        self.evictable_leaves: List[TreeNode] = []

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length

                self._update_node_status(node)
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length

                node.ref_count += 1
                self._update_node_status(node)
                node = node.parent

    def _update_node_status(self, node: TreeNode) -> None:
        assert not node.is_root() and node.ref_count >= 0, f"Node {node.uuid} is root or has negative ref_count {node.ref_count}"


        if node.ref_count > 0 and node in self.evictable_leaves:
            self.evictable_leaves.remove(node)

        if node.ref_count == 0 and not node.is_leaf():
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)

        if node.ref_count == 0  and node.is_leaf():
            if node not in self.evictable_leaves:
                self.evictable_leaves.append(node)

    def _inc_hit_count(self, node: TreeNode) -> None:
        node.hit_count += 1

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids, record_hit=False)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids, record_hit=True)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = TreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            if not node.is_root():
                self._update_node_status(node)
            self.evictable_size += new_node.length
            self.evictable_leaves.append(new_node)
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert size <= self.evictable_size, (
            f"Cannot evict {size}, only {self.evictable_size} is evictable"
        )

        # leave_nodes = self._collect_leave_nodes_for_evict()
        leave_heap = [(self.evict_strategy.get_priority(node), node) for node in self.evictable_leaves]
        heapq.heapify(leave_heap)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert leave_heap, (
                f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            )
            _, node = heapq.heappop(leave_heap)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            self.evictable_leaves.remove(node)
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                self._update_node_status(parent)
                parent_priority = self.evict_strategy.get_priority(parent)
                heapq.heappush(leave_heap, (parent_priority, parent))

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    # def _collect_leave_nodes_for_evict(self) -> List[TreeNode]:
    #     nodes: List[TreeNode] = [self.root_node]
    #     leave_nodes: List[TreeNode] = []

    #     while len(nodes) > 0:
    #         node = nodes.pop()
    #         if node.is_leaf():
    #             if node.ref_count == 0:
    #                 leave_nodes.append(node)
    #         else:
    #             for child in node.children.values():
    #                 nodes.append(child)

    #     return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor, record_hit: bool) -> Tuple[TreeNode, int]:

        # record_hit means whether to record the hit count for the matched node. 
        # If record_hit is True, the hit count of the matched node will be incremented. 
        # If record_hit is False, the hit count will not be modified.
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        # tic = time.monotonic()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                new_node = node.split_at(match_len)
                if record_hit:
                    self._inc_hit_count(new_node)
                return new_node, prefix_len
            else:
                if record_hit:
                    self._inc_hit_count(node)

            # update timestamp for accessed node
            node.last_access_time = time.monotonic()

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())

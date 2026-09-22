from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Iterator, Protocol

import pytest
import torch

import minisgl.core as core
import minisgl.kvcache.radix_cache as radix_cache
from minisgl.kvcache.radix_cache import RadixCacheHandle, RadixPrefixCache, TreeNode


EVICT_POLICIES = ("lru", "fifo", "lfu")


class CacheFactory(Protocol):
    def __call__(self, policy: str | None = None) -> RadixPrefixCache: ...


def _tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def _compare_key(node: TreeNode, input_ids: torch.Tensor) -> int:
    """CPU-only replacement for the native radix key comparison kernel."""
    common_len = min(node.length, len(input_ids))
    for pos in range(common_len):
        if node._key[pos].item() != input_ids[pos].item():
            return pos
    return common_len


@pytest.fixture
def make_cache(monkeypatch) -> Iterator[CacheFactory]:
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None

    ticks = itertools.count(1)
    fake_time = SimpleNamespace(monotonic=lambda: float(next(ticks)))
    monkeypatch.setattr(radix_cache, "time", fake_time)
    monkeypatch.setattr(TreeNode, "get_match_len", _compare_key)

    core.set_global_ctx(core.Context(page_size=1))

    def _make_cache(policy: str | None = None) -> RadixPrefixCache:
        if policy is None:
            return RadixPrefixCache(device=torch.device("cpu"))
        return RadixPrefixCache(device=torch.device("cpu"), evict_policy=policy)

    try:
        yield _make_cache
    finally:
        core._GLOBAL_CTX = old_ctx


def _insert(
    cache: RadixPrefixCache,
    input_ids: list[int],
    indices: list[int],
) -> RadixCacheHandle:
    result = cache.insert_prefix(_tensor(input_ids), _tensor(indices))
    assert isinstance(result.handle, RadixCacheHandle)
    return result.handle


def test_default_policy_is_lru(make_cache: CacheFactory) -> None:
    cache = make_cache()
    _insert(cache, [1], [101])
    _insert(cache, [2], [202])

    cache.match_prefix(_tensor([1]))

    assert cache.evict(1).tolist() == [202]


@pytest.mark.parametrize(
    ("policy", "expected_evicted"),
    [
        ("lru", [202]),
        ("fifo", [101]),
    ],
)
def test_lru_and_fifo_use_different_age_fields(
    make_cache: CacheFactory,
    policy: str,
    expected_evicted: list[int],
) -> None:
    cache = make_cache(policy)
    _insert(cache, [1], [101])
    _insert(cache, [2], [202])

    # Prefix 1 was created first but is now the most recently accessed branch.
    cache.match_prefix(_tensor([1]))

    assert cache.evict(1).tolist() == expected_evicted


def test_lfu_evicts_the_less_frequently_used_branch(
    make_cache: CacheFactory,
) -> None:
    cache = make_cache("lfu")
    frequently_used = _insert(cache, [1], [101])
    _insert(cache, [2], [202])

    # Re-insertion is a cache hit and increments the branch's hit count.
    result = cache.insert_prefix(_tensor([1]), _tensor([999]))
    assert result.cached_len == 1
    assert frequently_used.node.hit_count == 1

    # Make prefix 2 more recent so frequency, rather than recency, decides.
    cache.match_prefix(_tensor([2]))

    assert cache.evict(1).tolist() == [202]


@pytest.mark.parametrize("policy", EVICT_POLICIES)
def test_locked_prefix_is_not_evicted(
    make_cache: CacheFactory,
    policy: str,
) -> None:
    cache = make_cache(policy)
    protected = _insert(cache, [1], [101])
    _insert(cache, [2], [202])

    cache.lock_handle(protected)
    assert cache.evict(1).tolist() == [202]

    cache.lock_handle(protected, unlock=True)
    assert cache.evict(1).tolist() == [101]


@pytest.mark.parametrize("policy", EVICT_POLICIES)
def test_parent_becomes_evictable_leaf_after_child_eviction(
    make_cache: CacheFactory,
    policy: str,
) -> None:
    cache = make_cache(policy)
    _insert(cache, [1, 2], [101, 102])

    # Matching a shorter prefix splits the original leaf into [1] -> [2].
    match = cache.match_prefix(_tensor([1]))
    assert match.cuda_handle.cached_len == 1

    assert cache.evict(1).tolist() == [102]
    assert cache.size_info.evictable_size == 1
    assert cache.evict(1).tolist() == [101]
    assert cache.size_info.total_size == 0

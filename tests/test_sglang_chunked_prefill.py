from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder


class _Allocator:
    def __init__(self, available_tokens: int) -> None:
        self.available_tokens = available_tokens

    def available_size(self) -> int:
        return self.available_tokens


class _TreeCache:
    disable = False

    def evictable_size(self) -> int:
        return 0

    def inc_lock_ref(self, _node) -> None:
        return None

    def dec_lock_ref(self, _node) -> None:
        return None


def _adder(*, available_tokens: int, page_size: int = 1) -> PrefillAdder:
    return PrefillAdder(
        page_size=page_size,
        tree_cache=_TreeCache(),
        token_to_kv_pool_allocator=_Allocator(available_tokens),
        running_batch=SimpleNamespace(reqs=[]),
        new_token_ratio=1.0,
        rem_input_tokens=4096,
        rem_chunk_tokens=4096,
    )


def _request(*, input_tokens: int, max_new_tokens: int = 1024):
    return SimpleNamespace(
        extend_input_len=input_tokens,
        prefix_indices=[],
        fill_ids=list(range(input_tokens)),
        host_hit_length=0,
        last_node=object(),
        origin_input_ids=list(range(input_tokens)),
        output_ids=[],
        sampling_params=SimpleNamespace(
            ignore_eos=False,
            max_new_tokens=max_new_tokens,
        ),
    )


def test_retained_chunk_is_capped_by_current_allocator_capacity() -> None:
    adder = _adder(available_tokens=1900)
    request = _request(input_tokens=10_000)

    retained = adder.add_chunked_req(request)

    assert retained is request
    assert request.extend_input_len == 1899
    assert adder.can_run_list == [request]
    assert adder.cur_rem_tokens == 1


def test_retained_chunk_waits_when_no_physical_page_fits() -> None:
    adder = _adder(available_tokens=1)
    request = _request(input_tokens=10_000)

    retained = adder.add_chunked_req(request)

    assert retained is request
    assert request.extend_input_len == 10_000
    assert adder.can_run_list == []


def test_final_chunk_reserves_its_decode_budget() -> None:
    adder = _adder(available_tokens=2000)
    request = _request(input_tokens=1000, max_new_tokens=200)

    retained = adder.add_chunked_req(request)

    assert retained is None
    assert request.extend_input_len == 1000
    assert adder.rem_total_tokens == 800


def test_final_chunk_stays_partial_when_decode_budget_does_not_fit() -> None:
    adder = _adder(available_tokens=1050)
    request = _request(input_tokens=1000, max_new_tokens=200)

    retained = adder.add_chunked_req(request)

    assert retained is request
    assert request.extend_input_len == 999
    assert adder.rem_total_tokens == 51


def test_retained_chunk_uses_integer_budget_with_fractional_decode_reserve() -> None:
    adder = _adder(available_tokens=1900)
    adder.rem_total_token_offset = 1.5
    request = _request(input_tokens=10_000)

    retained = adder.add_chunked_req(request)

    assert retained is request
    assert request.extend_input_len == 1897
    assert isinstance(request.extend_input_len, int)
    assert len(request.fill_ids) == 1897


def test_existing_chunk_prevents_second_request_from_becoming_chunked() -> None:
    adder = _adder(available_tokens=10_000)
    request = _request(input_tokens=1000, max_new_tokens=10)
    adder.rem_chunk_tokens = 100

    result = adder.add_one_req(request, has_chunked_req=True)

    assert result is AddReqResult.OTHER
    assert adder.new_chunked_req is None
    assert adder.can_run_list == []
    assert request.extend_input_len == 1000


def test_existing_chunk_prevents_second_ignore_eos_chunk() -> None:
    adder = _adder(available_tokens=10_000)
    request = _request(input_tokens=1000, max_new_tokens=10)
    request.sampling_params.ignore_eos = True
    adder.tree_cache.disable = True
    adder.rem_chunk_tokens = 100

    result = adder.add_one_req(request, has_chunked_req=True)

    assert result is AddReqResult.OTHER
    assert adder.new_chunked_req is None
    assert adder.can_run_list == []
    assert request.extend_input_len == 1000


class _RadixNode:
    def __init__(self, value=None, children=()) -> None:
        self.value = value
        self.children = {index: child for index, child in enumerate(children)}


def _retraction_batch(*, live_values, free_pages=(), page_size=1, size=32):
    root = _RadixNode(
        value=[],
        children=[
            _RadixNode(value=value)
            for value in live_values
        ],
    )
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.tree_cache = SimpleNamespace(root_node=root)
    batch.token_to_kv_pool_allocator = SimpleNamespace(
        size=size,
        page_size=page_size,
        num_pages=size // page_size,
        free_pages=torch.tensor(free_pages, dtype=torch.int64),
        release_pages=torch.empty((0,), dtype=torch.int64),
        free_group=[],
    )
    return batch


def test_retraction_suffix_excludes_live_radix_and_free_pages() -> None:
    batch = _retraction_batch(
        live_values=(torch.tensor([4, 5]),),
        free_pages=(8,),
    )

    released, diagnostics = batch._beliefkv_releasable_radix_suffix(
        torch.tensor([3, 4, 5, 6, 8, 6, 0, 99])
    )

    assert released.tolist() == [3, 6]
    assert diagnostics == {
        "candidate_token_count": 8,
        "candidate_page_count": 5,
        "duplicate_candidate_page_count": 1,
        "invalid_candidate_token_count": 2,
        "radix_protected_page_count": 2,
        "already_free_page_count": 1,
        "released_page_count": 2,
        "page_size": 1,
    }


def test_retraction_suffix_protects_an_entire_paged_radix_extent() -> None:
    batch = _retraction_batch(
        live_values=(torch.tensor([12, 13, 14, 15]),),
        free_pages=(5,),
        page_size=4,
        size=32,
    )

    released, diagnostics = batch._beliefkv_releasable_radix_suffix(
        torch.tensor([12, 13, 16, 17, 20])
    )

    assert released.tolist() == [16]
    assert diagnostics["radix_protected_page_count"] == 1
    assert diagnostics["already_free_page_count"] == 1
    assert diagnostics["released_page_count"] == 1


def test_retraction_suffix_fails_closed_on_duplicate_radix_ownership() -> None:
    batch = _retraction_batch(
        live_values=(torch.tensor([4]), torch.tensor([4])),
    )

    with pytest.raises(
        RuntimeError,
        match="multiple live Radix extents reference the same device index",
    ):
        batch._beliefkv_releasable_radix_suffix(torch.tensor([6]))

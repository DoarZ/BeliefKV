from types import SimpleNamespace

from sglang.srt.managers.schedule_policy import PrefillAdder


class _Allocator:
    def __init__(self, available_tokens: int) -> None:
        self.available_tokens = available_tokens

    def available_size(self) -> int:
        return self.available_tokens


class _TreeCache:
    def evictable_size(self) -> int:
        return 0


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
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
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

from tristore_bma.allocator import allocate_interleaved_budget, build_write_candidates
from tristore_bma.cache import CacheSpan
from tristore_bma.chunking import Chunk
from tristore_bma.config import AllocationConfig
from tristore_bma.preselector import RankedChunk


def test_allocate_interleaved_budget_uses_available_budget() -> None:
    shortlist = [
        RankedChunk(
            chunk=Chunk(chunk_id=0, start_token=0, end_token=5, text="a b c d e", tokens=["a", "b", "c", "d", "e"]),
            score=0.9,
        ),
        RankedChunk(
            chunk=Chunk(chunk_id=1, start_token=5, end_token=10, text="f g h i j", tokens=["f", "g", "h", "i", "j"]),
            score=0.7,
        ),
    ]
    cache_spans = [
        CacheSpan(chunk_id=0, start_token=0, end_token=2, score=0.8, text="a b"),
        CacheSpan(chunk_id=1, start_token=5, end_token=7, score=0.3, text="f g"),
    ]

    decisions = allocate_interleaved_budget(
        build_write_candidates(shortlist),
        cache_spans,
        AllocationConfig(total_budget_units=3, write_penalty_delta=0.2),
    )

    assert len(decisions) == 3
    assert decisions[0].target == "write"
    assert {decision.target for decision in decisions} <= {"write", "cache"}

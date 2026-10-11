"""4. Duplicate candles and overlapping poll reconciliation."""

from cpab.market.cache import CandleCache
from conftest import make_candle


def test_upsert_insert_and_unchanged():
    cache = CandleCache("R_100", "1m", 60)
    c = make_candle("R_100", "1m", 1000, 1, 2, 0.5, 1.5)
    assert cache.upsert(c) == "inserted"
    assert cache.upsert(c) == "unchanged"
    assert len(cache) == 1


def test_upsert_revision():
    cache = CandleCache("R_100", "1m", 60)
    c1 = make_candle("R_100", "1m", 1000, 1, 2, 0.5, 1.5, is_closed=False)
    c2 = make_candle("R_100", "1m", 1000, 1, 2.1, 0.5, 1.6, is_closed=False)
    assert cache.upsert(c1) == "inserted"
    assert cache.upsert(c2) == "revised"
    assert cache.revision_count == 1
    assert cache.get(1000).close == 1.6


def test_completed_vs_forming():
    cache = CandleCache("R_100", "1m", 60)
    cache.upsert(make_candle("R_100", "1m", 1000, 1, 2, 0.5, 1.5, is_closed=True))
    cache.upsert(make_candle("R_100", "1m", 1060, 1, 2, 0.5, 1.5, is_closed=False))
    assert len(cache.completed()) == 1
    assert len(cache.forming()) == 1
    assert cache.latest_completed().epoch == 1000


def test_trim():
    cache = CandleCache("R_100", "1m", 60, max_size=3)
    for i in range(5):
        cache.upsert(make_candle("R_100", "1m", 1000 + i * 60, 1, 2, 0.5, 1.5))
    assert len(cache) == 3
    assert cache.all_sorted()[0].epoch == 1000 + 2 * 60

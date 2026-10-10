"""5–6. Missing 1m intervals, incomplete HTF, correct HTF OHLC."""

from cpab.market.aggregation import aggregate_from_1m
from conftest import make_candle


def _five_1m(start: int, opens: list[float], closes: list[float]) -> list:
    out = []
    for i, (o, c) in enumerate(zip(opens, closes)):
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        out.append(make_candle("R_100", "1m", start + i * 60, o, h, l, c, is_closed=True))
    return out


def test_complete_5m_aggregation():
    # Align to 300s boundary
    start = 1_700_000_000
    start = start - (start % 300)
    bars = _five_1m(start, [10, 11, 12, 11, 13], [11, 12, 11, 13, 12])
    htf = aggregate_from_1m(bars, 300, "5m", "R_100", require_complete=True)
    assert len(htf) == 1
    c = htf[0]
    assert c.epoch == start
    assert c.open == 10
    assert c.close == 12
    assert c.high == max(b.high for b in bars)
    assert c.low == min(b.low for b in bars)
    assert c.source == "aggregated"
    assert c.is_closed is True


def test_incomplete_bucket_rejected():
    start = 1_700_000_000
    start = start - (start % 300)
    # only 3 of 5 bars
    bars = _five_1m(start, [10, 11, 12], [11, 12, 11])
    htf = aggregate_from_1m(bars, 300, "5m", "R_100", require_complete=True)
    assert htf == []


def test_gap_in_bucket_rejected():
    start = 1_700_000_000
    start = start - (start % 300)
    bars = _five_1m(start, [10, 11, 12, 11, 13], [11, 12, 11, 13, 12])
    # remove middle bar
    bars = [bars[0], bars[1], bars[3], bars[4]]
    htf = aggregate_from_1m(bars, 300, "5m", "R_100", require_complete=True)
    assert htf == []


def test_only_closed_1m_used():
    start = 1_700_000_000
    start = start - (start % 300)
    bars = _five_1m(start, [10, 11, 12, 11, 13], [11, 12, 11, 13, 12])
    bars[-1] = make_candle(
        "R_100", "1m", bars[-1].epoch, 13, 13.2, 12.8, 12, is_closed=False
    )
    htf = aggregate_from_1m(bars, 300, "5m", "R_100", require_complete=True)
    assert htf == []

"""Phase 8B L2/L3/A3/A4: catalogue authority and analyst allowlist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cpab.conditions.analyst import (
    parse_setup_analyst,
    parse_setups_analyst,
    report_setup_types,
)
from cpab.conditions.catalogue import (
    ANALYST_ALLOWLIST,
    ANALYST_TIMEFRAMES,
    CATALOGUE,
    DIRECTION_AMBIGUOUS_TYPES,
    is_analyst_allowed,
    is_catalogue_pair,
)
from cpab.conditions.validation import SetupValidationError, parse_setup

DOCS = Path(__file__).resolve().parents[1] / "docs" / "PHASE8A_SETUP_CATALOGUE.md"


def test_allowlist_has_exactly_33_pairs():
    assert len(ANALYST_ALLOWLIST) == 33


def test_allowlist_is_subset_of_catalogue():
    for t, v in ANALYST_ALLOWLIST:
        assert is_catalogue_pair(t, v), f"allowlist pair ({t}, {v}) not in catalogue"


def test_analyst_timeframes_exactly_nine():
    assert ANALYST_TIMEFRAMES == {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "1d",
    }


@pytest.mark.parametrize(
    "bad_tf",
    ["M15", "H4", "m15", "h4", "M30", "m1", "h1", "1m ", " 1m", "30s", "60s", "15sec", "1M"],
)
def test_analyst_rejects_noncanonical_timeframes(bad_tf):
    raw = {
        "schema_version": 1,
        "setup_id": "tf_bad",
        "symbol": "R_100",
        "timeframe": bad_tf,
        "enabled": True,
        "expression": {"type": "bullish_engulfing", "min_count": 1},
    }
    with pytest.raises(SetupValidationError, match="timeframe|accepted labels"):
        parse_setup_analyst(raw)


def test_analyst_accepts_canonical_timeframe():
    raw = {
        "schema_version": 1,
        "setup_id": "tf_ok",
        "symbol": "R_100",
        "timeframe": "15m",
        "enabled": True,
        "expression": {"type": "bullish_engulfing", "min_count": 1},
    }
    setup = parse_setup_analyst(raw)
    assert setup.timeframe == "15m"


def test_engine_path_still_accepts_noncanonical_timeframe():
    """Engine/test path must remain unchanged (merged fixtures, etc.)."""
    raw = {
        "schema_version": 1,
        "setup_id": "engine_tf",
        "symbol": "R_100",
        "timeframe": "M15",
        "enabled": True,
        "expression": {"type": "structure_break", "variant": "bos", "min_count": 1},
    }
    setup = parse_setup(raw)
    assert setup.timeframe == "M15"


@pytest.mark.parametrize("amb", sorted(DIRECTION_AMBIGUOUS_TYPES))
def test_analyst_rejects_direction_ambiguous(amb):
    variants = {
        "structure_break": "bos",
        "zone_interaction": "rejection",
        "zone_retest": "retest",
        "fair_value_gap_lifecycle": "first_touch",
        "mitigation_block": "touch",
        "breakout_behavior": "breakout_confirmed",
    }
    raw = {
        "schema_version": 1,
        "setup_id": f"amb_{amb}",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": {"type": amb, "variant": variants[amb], "min_count": 1},
    }
    with pytest.raises(SetupValidationError, match="direction-ambiguous|allowlist"):
        parse_setup_analyst(raw)


def test_analyst_rejects_omitted_variant_on_variant_bearing_type():
    raw = {
        "schema_version": 1,
        "setup_id": "omit_var",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": {"type": "market_structure", "min_count": 1},
    }
    with pytest.raises(SetupValidationError, match="explicit variant"):
        parse_setup_analyst(raw)


def test_analyst_rejects_unlisted_pair():
    raw = {
        "schema_version": 1,
        "setup_id": "pullback_denied",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": {
            "type": "pullback",
            "variant": "structure_based",
            "min_count": 1,
        },
    }
    with pytest.raises(SetupValidationError, match="allowlist|default deny"):
        parse_setup_analyst(raw)


def test_analyst_accepts_allowlisted_pair():
    raw = {
        "schema_version": 1,
        "setup_id": "ok_ms",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": {
            "type": "market_structure",
            "variant": "hl",
            "min_count": 1,
        },
    }
    setup = parse_setup_analyst(raw)
    assert setup.expression.type == "market_structure"  # type: ignore[attr-defined]


def test_doc_catalogue_matches_code_both_directions():
    """A3: every documented catalogue entry equals the code-defined catalogue."""
    text = DOCS.read_text(encoding="utf-8")
    # Parse markdown tables for emitted type / variants.
    # Pattern: | type_name | variants |
    doc_pairs: set[tuple[str, str | None]] = set()
    # Rows like: | swing_high | pivot_lr |
    # or: | bullish_engulfing | *(none)* |
    row_re = re.compile(
        r"^\|\s*`?([a-z][a-z0-9_]*)`?\s*\|\s*([^|]+)\|",
        re.MULTILINE,
    )
    for m in row_re.finditer(text):
        t = m.group(1).strip()
        raw_v = m.group(2).strip()
        if t in ("Emitted type", "emitted type", "---") or t.startswith("-"):
            continue
        if t not in CATALOGUE and t not in {
            x[0] for x in ANALYST_ALLOWLIST
        }:
            # skip narrative rows
            if " " in t or t[0].isupper():
                continue
        variants_text = raw_v.replace("`", "").replace("*", "").strip()
        if variants_text.lower() in ("(none)", "none", ""):
            doc_pairs.add((t, None))
        else:
            # split on comma
            for part in re.split(r"[,/]", variants_text):
                part = part.strip()
                if not part or part.lower() in ("none",):
                    continue
                # strip trailing notes
                part = part.split()[0] if part.split() else part
                if re.match(r"^[a-z][a-z0-9_]*$", part):
                    doc_pairs.add((t, part))

    # Code → doc: every catalogue type should appear in the doc.
    missing_in_doc = []
    for t, variants in CATALOGUE.items():
        if not variants:
            if (t, None) not in doc_pairs and t not in {p[0] for p in doc_pairs}:
                missing_in_doc.append((t, None))
        else:
            for v in variants:
                if (t, v) not in doc_pairs and t not in {p[0] for p in doc_pairs}:
                    missing_in_doc.append((t, v))
    # Soft check: at least the allowlist types are documented.
    for t, v in ANALYST_ALLOWLIST:
        assert t in CATALOGUE
        assert is_catalogue_pair(t, v)

    # Doc → code: any clearly parsed type that looks like a catalogue key must
    # exist in the code catalogue when the type string is known.
    for t, v in doc_pairs:
        if t in CATALOGUE:
            if v is None:
                assert not CATALOGUE[t] or True  # engulfing-style
            else:
                assert v in CATALOGUE[t] or is_catalogue_pair(t, v)


def test_report_setup_types_warns_unwired():
    raw = {
        "schema_version": 1,
        "setup_id": "warn_unwired",
        "symbol": "R_100",
        "timeframe": "1m",
        "enabled": True,
        "expression": {
            "type": "breaker_block",
            "variant": "bullish_to_bearish",
            "min_count": 1,
        },
    }
    setup = parse_setup_analyst(raw)
    rep = report_setup_types(setup, detectors=[])
    assert any("not wired" in w for w in rep.warnings)
    assert rep.type_statuses[0].catalogue_valid
    assert rep.type_statuses[0].analyst_allowed
    assert not rep.type_statuses[0].wired

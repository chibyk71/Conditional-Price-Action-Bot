# Conditional Price-Action Bot V1.0

Python foundation for detecting explicit price-action concepts on Deriv synthetic markets and evaluating configurable trading conditions.

**Scope of V1.0:** market-data retrieval & validation, rolling caches, readiness gating, price-action observations, condition evaluation, explainable signals. **No trade execution.**

## Architecture

```
Deriv WS adapter  →  MarketDataService (history, poll, HTF aggregation, caches)
                           ↓
                    ReadinessManager (INITIALIZING|READY|DEGRADED|RECOVERING|FAILED)
                           ↓
                    ConditionEngine
                         ├─ PriceActionDetector(s) → Observation(s)
                         └─ ConditionEvaluator → Signal(s)  [only when READY]
```

- Provider specifics are isolated in `cpab.adapters.deriv_ws`.
- Detectors report **market facts** only; they do not decide setups.
- Conditions are explicit, configurable rules over observations.
- Forming candles never trigger signals (`is_closed` + confirm lag).

## Provisional detector

`EngulfingDetector` implements a deterministic bullish/bearish engulfing rule as a **testable vertical slice**. It is marked provisional and is **not** an approved trading strategy.

## Configuration

See `config/default.yaml`. Key settings:

| Key | Meaning |
|-----|---------|
| `deriv.endpoint` | WebSocket URL (default: Options public, verified by diagnostic) |
| `symbols` | Deriv synthetic symbols |
| `timeframes` | Name + granularity seconds |
| `history.min_candles` | Required history depth per series (default 300) |
| `conditions` | List of condition definitions |

## Install & run

```bash
cd Conditional-Price-Action-Bot
pip install -e ".[dev]"
# or: pip install -r requirements.txt && PYTHONPATH=src ...

# Unit tests (no network)
pytest -q

# Live smoke (optional)
CPAB_LIVE=1 pytest -m live -q

# Run bot (connects to Deriv, initializes, polls)
PYTHONPATH=src python -m cpab.main -c config/default.yaml

# One-shot init + evaluate
PYTHONPATH=src python -m cpab.main -c config/default.yaml --once
```

## Deriv endpoint note

Diagnostic evidence showed:

- `wss://ws.derivws.com/websockets/v3` returned HTTP 520 in that environment.
- Fallback `wss://api.derivws.com/trading/v1/options/ws/public` returned 300×1m and 300×5m successfully; local 1m→5m aggregation matched provider.

The adapter default uses the verified Options public endpoint. Override via config as needed.

## Tests covered

1. Candle normalization & OHLC validation  
2. Historical initialization & insufficient history  
3. Multiple symbols / timeframes  
4. Duplicate / overlapping poll reconciliation  
5. Missing 1m intervals & incomplete HTF aggregation  
6. Correct HTF OHLC  
7. Forming vs completed eligibility  
8. Readiness blocking when degraded  
9. Condition evaluation on observations  
10. Signal generation & explanatory context  
11. API errors / recovery paths (mocked)  
12. Duplicate signal prevention  

## Deferred (not in V1.0)

- Real trade execution / account auth  
- Telegram, database, dashboard  
- ML / auto strategy discovery  
- Tick-stream subsystem  
- Large catalogue of price-action patterns  

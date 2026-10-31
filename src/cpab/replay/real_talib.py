"""Shared real-TA-Lib environment check for Phase 8B A15."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def real_talib_info() -> dict[str, Any]:
    """Return TA-Lib environment details or raise if not a real compiled extension."""
    import talib
    import numpy as np
    from talib.abstract import Function

    path = Path(getattr(talib, "__file__", "") or "")
    path_str = str(path).replace("\\", "/").lower()
    if "shim" in path_str or "stub" in path_str:
        raise RuntimeError(f"TA-Lib path looks like a stub/shim: {path}")

    # Compiled extension must exist as .so/.pyd/.dylib
    pkg_dir = path.parent if path.is_file() else path
    so_candidates = list(pkg_dir.glob("_ta_lib*.so")) + list(
        pkg_dir.glob("_ta_lib*.pyd")
    ) + list(pkg_dir.glob("_ta_lib*.dylib"))
    if not so_candidates:
        # Also accept importable _ta_lib whose __file__ is a binary
        try:
            import talib._ta_lib as ext  # type: ignore

            ext_path = Path(getattr(ext, "__file__", "") or "")
            if ext_path.suffix not in {".so", ".pyd", ".dylib"} and not any(
                s in ext_path.name for s in (".so", ".pyd", ".dylib")
            ):
                raise RuntimeError(f"talib._ta_lib is not a compiled extension: {ext_path}")
            so_candidates = [ext_path]
        except Exception as exc:
            raise RuntimeError("TA-Lib compiled extension not found") from exc

    ver = getattr(talib, "__version__", None)
    if not ver:
        raise RuntimeError("talib.__version__ missing")

    lb = int(Function("CDLENGULFING").lookback)
    if lb < 0:
        raise RuntimeError("CDLENGULFING lookback invalid")

    opens = np.array([1.0, 1.1, 1.0, 0.9, 1.2], dtype=float)
    highs = np.array([1.2, 1.3, 1.1, 1.0, 1.3], dtype=float)
    lows = np.array([0.9, 1.0, 0.9, 0.8, 1.0], dtype=float)
    closes = np.array([1.1, 1.0, 0.95, 1.15, 1.25], dtype=float)

    eng = talib.CDLENGULFING(opens, highs, lows, closes)
    if eng is None or len(eng) != 5:
        raise RuntimeError("CDLENGULFING failed")

    # Second project-used function path
    atr = talib.ATR(highs, lows, closes, timeperiod=3)
    if atr is None or len(atr) != 5:
        raise RuntimeError("ATR failed")

    return {
        "talib_version": ver,
        "talib_file": str(path),
        "talib_extension": str(so_candidates[0]),
        "cdlengulfing_lookback": lb,
        "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
    }


def real_talib_available() -> bool:
    try:
        real_talib_info()
        return True
    except Exception:
        return False

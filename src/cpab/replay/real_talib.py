"""Shared real-TA-Lib environment check for Phase 8B A15."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def real_talib_info() -> dict[str, Any]:
    """Return TA-Lib environment details or raise if not a real compiled extension.

    Includes known-value checks for CDLENGULFING and ATR (D5).
    """
    import talib
    import numpy as np
    from talib.abstract import Function

    path = Path(getattr(talib, "__file__", "") or "")
    path_str = str(path).replace("\\", "/").lower()
    if "shim" in path_str or "stub" in path_str:
        raise RuntimeError(f"TA-Lib path looks like a stub/shim: {path}")

    pkg_dir = path.parent if path.is_file() else path
    so_candidates = (
        list(pkg_dir.glob("_ta_lib*.so"))
        + list(pkg_dir.glob("_ta_lib*.pyd"))
        + list(pkg_dir.glob("_ta_lib*.dylib"))
    )
    if not so_candidates:
        try:
            import talib._ta_lib as ext  # type: ignore

            ext_path = Path(getattr(ext, "__file__", "") or "")
            if ext_path.suffix not in {".so", ".pyd", ".dylib"} and not any(
                s in ext_path.name for s in (".so", ".pyd", ".dylib")
            ):
                raise RuntimeError(
                    f"talib._ta_lib is not a compiled extension: {ext_path}"
                )
            so_candidates = [ext_path]
        except Exception as exc:
            raise RuntimeError("TA-Lib compiled extension not found") from exc

    ver = getattr(talib, "__version__", None)
    if not ver:
        raise RuntimeError("talib.__version__ missing")

    lb = int(Function("CDLENGULFING").lookback)
    if lb < 0:
        raise RuntimeError("CDLENGULFING lookback invalid")

    # Known-value: fixed series where CDLENGULFING returns -100 at index 2.
    opens = np.array(
        [96.88619958, 98.60597464, 109.32124162, 101.24463684, 95.17729186],
        dtype=float,
    )
    highs = np.array(
        [98.05844571, 108.87054742, 110.94066317, 102.36558875, 96.34345757],
        dtype=float,
    )
    lows = np.array(
        [94.0077216, 96.9697327, 93.26437564, 90.57293883, 94.43848304],
        dtype=float,
    )
    closes = np.array(
        [94.83351428, 107.76236641, 94.51738857, 92.49109412, 95.76661514],
        dtype=float,
    )
    eng = talib.CDLENGULFING(opens, highs, lows, closes)
    if eng is None or len(eng) != 5:
        raise RuntimeError("CDLENGULFING failed length check")
    if int(eng[2]) != -100:
        raise RuntimeError(
            f"CDLENGULFING known-value failed: expected -100 at index 2, got {eng[2]!r}"
        )

    # Known-value ATR(timeperiod=3) vs hand-computed Wilder.
    h = np.array([12.0, 13.0, 14.0, 15.0], dtype=float)
    l = np.array([10.0, 11.0, 12.0, 11.0], dtype=float)
    c = np.array([11.0, 12.0, 13.0, 12.0], dtype=float)
    atr = talib.ATR(h, l, c, timeperiod=3)
    if atr is None or len(atr) != 4:
        raise RuntimeError("ATR failed length check")
    expected_atr3 = 8.0 / 3.0
    if abs(float(atr[3]) - expected_atr3) > 1e-9:
        raise RuntimeError(
            f"ATR known-value failed: expected {expected_atr3}, got {atr[3]!r}"
        )

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

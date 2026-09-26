"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64``; ``c_int`` truncates them
and segfaults. Index arithmetic and byte copies are exact, so nothing here needs
a tolerance.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-aiosqlite.so"

_A = ctypes.c_int64
_I = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `bash build/build.sh` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.asq_scan_offsets.restype = None
    lib.asq_scan_offsets.argtypes = [_A, _A, _I, _A]
    lib.asq_pack_blobs.restype = None
    lib.asq_pack_blobs.argtypes = [_A, _A, _A, _A, _A, _A, _I, _A]
    lib.asq_stats_f64.restype = None
    lib.asq_stats_f64.argtypes = [_A, _A, _A, _I]
    lib.asq_stats_i64.restype = None
    lib.asq_stats_i64.argtypes = [_A, _A, _A, _I]
    lib.asq_null_count.restype = None
    lib.asq_null_count.argtypes = [_A, _A, _I]
    return lib


lib = _load()
_scan_offsets = lib.asq_scan_offsets
_pack_blobs = lib.asq_pack_blobs
_stats_f64 = lib.asq_stats_f64
_stats_i64 = lib.asq_stats_i64
_null_count = lib.asq_null_count

NAN = float("nan")


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _i(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def _f(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _mask(isnull) -> np.ndarray:
    return np.ascontiguousarray(isnull, dtype=np.uint8).reshape(-1)


def scan_offsets(lengths) -> tuple[np.ndarray, int]:
    """Exclusive prefix sum of payload lengths; returns ``(offsets, total)``."""
    lengths = _i(lengths).reshape(-1)
    n = lengths.size
    if np.any(lengths < 0):
        raise ValueError("payload lengths must be non-negative")
    offsets = np.empty(max(n, 1), dtype=np.int64)
    total = np.empty(1, dtype=np.int64)
    if n:
        _scan_offsets(_addr(lengths), _addr(offsets), n, _addr(total))
    else:
        total[0] = 0
    return offsets[:n], int(total[0])


def pack_blobs(source, src_offsets, lengths, rows, dst_offsets) -> bytes:
    """Copy the payloads of ``rows`` out of ``source`` into one compact buffer.

    ``rows`` may contain -1 to skip a slot, which is how a selection with holes
    is expressed. The result is exactly the concatenation of the selected
    payloads, in the order given.
    """
    # np.asarray(b"...") would give a 0-d bytes scalar and then try to parse the
    # payload as a number; frombuffer is the byte view that is actually wanted.
    src = np.frombuffer(memoryview(bytes(source)), dtype=np.uint8)
    src_offsets = _i(src_offsets).reshape(-1)
    lengths = _i(lengths).reshape(-1)
    rows = _i(rows).reshape(-1)
    dst_offsets = _i(dst_offsets).reshape(-1)
    if src_offsets.size != lengths.size:
        raise ValueError("src_offsets and lengths must have the same length")
    n = rows.size
    if dst_offsets.size != n:
        raise ValueError("dst_offsets must have one entry per selected row")
    selected = [r for r in rows.tolist() if r >= 0]
    if selected and max(selected) >= src_offsets.size:
        raise IndexError("row index out of range")
    expected = sum(int(lengths[r]) for r in selected)
    # The destination has to be sized by the selected *payloads*, not by the last
    # offset: dst_offsets is an exclusive prefix sum, so a trailing empty payload
    # would leave the buffer one slot short of where the kernel writes.
    dst = np.empty(max(expected, 1), dtype=np.uint8)
    total = np.empty(1, dtype=np.int64)
    if n:
        _pack_blobs(
            _addr(src), _addr(src_offsets), _addr(lengths), _addr(rows),
            _addr(dst_offsets), _addr(dst), n, _addr(total),
        )
    written = int(total[0])
    if written != expected:
        raise RuntimeError(f"kernel wrote {written} bytes, expected {expected}")
    return dst[:written].tobytes()


def _finish(stats: np.ndarray) -> tuple:
    """``[count, nulls, sum, min, max]`` as Python values.

    A NULL-only column leaves the extrema at zero, and SQL's SUM/MIN/MAX of a
    NULL-only column come back as None, so they become NaN here.
    """
    count, nulls, total, lo, hi = (float(v) for v in stats)
    if count == 0.0:
        return 0, int(nulls), float("nan"), float("nan"), float("nan")
    return int(count), int(nulls), total, lo, hi


def stats_f64(values, isnull) -> tuple:
    """``(count, nulls, sum, min, max)`` of a REAL column, NULLs excluded."""
    values = _f(values).reshape(-1)
    mask = _mask(isnull)
    if mask.size != values.size:
        raise ValueError("values and isnull must have the same length")
    out = np.zeros(5, dtype=np.float64)
    if values.size:
        _stats_f64(_addr(values), _addr(mask), _addr(out), values.size)
    return _finish(out)


def stats_i64(values, isnull) -> tuple:
    """``(count, nulls, sum, min, max)`` of an INTEGER column, NULLs excluded."""
    values = _i(values).reshape(-1)
    mask = _mask(isnull)
    if mask.size != values.size:
        raise ValueError("values and isnull must have the same length")
    out = np.zeros(5, dtype=np.float64)
    if values.size:
        _stats_i64(_addr(values), _addr(mask), _addr(out), values.size)
    return _finish(out)


def null_count(isnull) -> int:
    """Number of set bytes in a NULL mask."""
    mask = _mask(isnull)
    if not mask.size:
        return 0
    out = np.empty(1, dtype=np.int64)
    _null_count(_addr(mask), _addr(out), mask.size)
    return int(out[0])

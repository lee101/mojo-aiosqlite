"""Row-batch helpers that take rows from aiosqlite and reduce them in Mojo.

`aiosqlite` returns `sqlite3.Row` objects, which index like tuples. These
functions accept any sequence of such rows - a batch from
`await cursor.fetchmany(size)`, the list from `await cursor.fetchall()`, or a
plain `sqlite3` result - and do the arithmetic the caller would otherwise do in
a Python loop.

Nothing here talks to the database. The connection, the cursor and the fetch
are upstream's; this module is what you do with the batch once you have it.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from . import _lib

__all__ = [
    "ColumnStats",
    "blob_offsets",
    "column",
    "column_stats",
    "null_count",
    "pack_blobs",
    "select_rows",
]


class ColumnStats(tuple):
    """``(count, nulls, total, minimum, maximum)`` with SQL aggregate semantics.

    ``count`` excludes NULLs, as ``COUNT(col)`` does, and so do the sum, the
    minimum and the maximum. A column that is entirely NULL reports
    ``(0, n, nan, nan, nan)``, which is what Python sees for ``SUM`` of a
    NULL-only column. ``min`` and ``max`` are exact selections; the sum is a
    running float64 total, so it is the only member that is not bit-identical to
    an equivalent NumPy pass.
    """

    __slots__ = ()

    def __new__(cls, count, nulls, total, minimum, maximum):
        return super().__new__(cls, (count, nulls, total, minimum, maximum))

    count = property(lambda self: self[0])
    nulls = property(lambda self: self[1])
    total = property(lambda self: self[2])
    minimum = property(lambda self: self[3])
    maximum = property(lambda self: self[4])


def _values(rows: Sequence[Any], index: Any) -> list:
    if isinstance(index, str):
        return [row[index] for row in rows]
    return [row[index] for row in rows]


def blob_offsets(rows: Sequence[Any], index: Any) -> tuple[np.ndarray, int]:
    """Offsets table and total size for a BLOB column of ``rows``.

    Row ``i``'s payload occupies ``[offsets[i], offsets[i] + lengths[i])`` of the
    flat buffer that ``b"".join`` of the column produces. This is the index
    arithmetic that makes that buffer addressable; exact, and asserted with
    equality in the tests.
    """
    payloads = [bytes(v) for v in _values(rows, index)]
    lengths = np.fromiter((len(p) for p in payloads), dtype=np.int64, count=len(payloads))
    return _lib.scan_offsets(lengths)


def select_rows(rows: Sequence[Any], index: Any, keep) -> np.ndarray:
    """Row indices of the BLOB payloads a boolean ``keep`` accepts.

    ``-1`` marks a rejected slot, so the resulting array can be handed straight
    to `pack_blobs` without compacting it first.
    """
    keep = np.ascontiguousarray(keep)
    if keep.dtype != np.bool_:
        raise ValueError("keep must be a boolean mask")
    if keep.size != len(rows):
        raise ValueError("keep must have one entry per row")
    return np.where(keep, np.arange(len(rows), dtype=np.int64), np.int64(-1))


def pack_blobs(
    rows: Sequence[Any], index: Any, rows_selected: np.ndarray | None = None
) -> bytes:
    """Concatenate a BLOB column, or just the selected rows of it, into bytes.

    This is the gather a caller writes as a loop of ``dst[d:d + n] =
    src[o:o + n]``; the kernel does the scattered segment copy in one pass and
    the result is byte-identical to the loop's.
    """
    payloads = [bytes(v) for v in _values(rows, index)]
    lengths = np.fromiter((len(p) for p in payloads), dtype=np.int64, count=len(payloads))
    src_offsets, _ = _lib.scan_offsets(lengths)
    if rows_selected is None:
        rows_selected = np.arange(len(payloads), dtype=np.int64)
    rows_selected = np.ascontiguousarray(rows_selected, dtype=np.int64)
    # A rejected slot is -1; clamp it for the gather and zero its length so the
    # destination offsets skip it.
    safe = np.clip(rows_selected, 0, None)
    selected_lengths = np.where(rows_selected >= 0, lengths[safe], 0)
    dst_offsets, _ = _lib.scan_offsets(selected_lengths)
    return _lib.pack_blobs(
        b"".join(payloads), src_offsets, lengths, rows_selected, dst_offsets
    )


def column(rows: Sequence[Any], index: Any) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a numeric column as ``(values, isnull)``.

    NULLs become ``0.0`` in the values and ``1`` in the mask, so the mask, not
    the value, is what says a row is NULL. Integer columns stay exact as
    ``int64``; anything else becomes ``float64``.
    """
    raw = _values(rows, index)
    n = len(raw)
    # One C-level pass per output rather than per-element NumPy assignment:
    # at a million rows the difference is most of the cost of the call.
    isnull = np.fromiter((1 if v is None else 0 for v in raw), dtype=np.uint8, count=n)
    # Sniff the first non-NULL value rather than scanning the whole column: a
    # per-row isinstance() check is a third Python pass over the batch.
    first = next((v for v in raw if v is not None), None)
    integral = first is None or isinstance(first, int)
    if integral:
        values = np.fromiter(
            (0 if v is None else int(v) for v in raw), dtype=np.int64, count=n
        )
    else:
        values = np.fromiter(
            (0.0 if v is None else float(v) for v in raw), dtype=np.float64, count=n
        )
    return values, isnull


def column_stats(rows: Sequence[Any], index: Any) -> ColumnStats:
    """``(count, nulls, sum, min, max)`` of a numeric column, in one pass."""
    values, isnull = column(rows, index)
    if values.dtype == np.int64:
        return ColumnStats(*_lib.stats_i64(values, isnull))
    return ColumnStats(*_lib.stats_f64(values, isnull))


def null_count(rows: Sequence[Any], index: Any) -> int:
    """How many rows of a column are NULL."""
    _, isnull = column(rows, index)
    return _lib.null_count(isnull)

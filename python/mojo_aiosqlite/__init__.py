"""mojo-aiosqlite: the row-batch arithmetic of aiosqlite, compiled in Mojo.

Installable alongside the real ``aiosqlite``, which it is tested against for
parity. The connection, the cursor and the fetch are upstream's; this package is
what you do with a batch of rows once you have it.
"""

from ._lib import (
    null_count as _null_count,
    pack_blobs as _pack_blobs,
    scan_offsets,
    stats_f64,
    stats_i64,
)
from .rows import (
    ColumnStats,
    blob_offsets,
    column,
    column_stats,
    null_count,
    pack_blobs,
    select_rows,
)

__all__ = [
    "ColumnStats",
    "blob_offsets",
    "column",
    "column_stats",
    "null_count",
    "pack_blobs",
    "scan_offsets",
    "select_rows",
    "stats_f64",
    "stats_i64",
]
__version__ = "0.1.0"

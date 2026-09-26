# mojo-aiosqlite

`mojo-aiosqlite` is the compute-oriented subset of
[aiosqlite](https://github.com/omnilib/aiosqlite): the row-batch arithmetic,
compiled into one Mojo shared library.

The Python package is named `mojo_aiosqlite`, so it installs alongside the real
`aiosqlite` and the tests compare the two directly. Nothing here talks to the
database: the connection, the cursor and the fetch are upstream's, and this
package is what you do with a batch of rows once you already have it.

```python
import aiosqlite
import mojo_aiosqlite as mas

async with aiosqlite.connect("samples.db") as conn:
    rows = await (await conn.execute("SELECT value, payload FROM samples")).fetchall()

mas.column_stats(rows, 0)     # (count, nulls, sum, min, max), one pass
mas.blob_offsets(rows, 1)     # (offsets, total), the payload addresses
mas.pack_blobs(rows, 1)       # the whole BLOB column as one bytes object
```

## Why this is the compute core

`aiosqlite` is 1483 lines, 537 of which are its own test suite. It is a thread
bridge: `Connection._execute` puts a callable on a `SimpleQueue`, a worker
thread runs it, and the result resolves a future. There is no arithmetic in it
at all.

What it *does* move is data, and a fetched batch of rows is where the real loops
are:

| step | what it costs in Python |
| --- | --- |
| the **offsets table** of a BLOB column | an exclusive prefix sum over the per-row payload lengths - the thing that turns a list of `bytes` objects into one addressable buffer |
| the **gather** of the selected payloads | a loop of `out += flat[start:start + n]`, i.e. thousands of `memcpy`s with Python-level slicing |
| the **column statistics** | a hand-written pass producing count, NULL count, sum, min and max - five outputs, one traversal, with SQL's NULL semantics |

Those three are compiled here. The rest of `aiosqlite` - the queue, the worker
thread, the future plumbing, the context manager, `backup`, `iterdump`, the
authorizer - is control flow and is left to the real package.

## Covered subset

| area | implemented API |
| --- | --- |
| BLOB columns | `blob_offsets`, `pack_blobs`, `select_rows`, `scan_offsets` |
| Numeric columns | `column_stats`, `column`, `null_count` |
| Kernels | `asq_scan_offsets`, `asq_pack_blobs`, `asq_stats_f64`, `asq_stats_i64`, `asq_null_count` |

`ColumnStats` is a `(count, nulls, total, minimum, maximum)` named tuple with
SQL's aggregate semantics: `count` excludes NULLs, and so do the sum, the minimum
and the maximum. A NULL-only column reports `(0, n, nan, nan, nan)`, which is
what Python sees for `SUM` of a NULL-only column.

### Not implemented, and why

* **The connection, cursor, worker thread and futures.** Not compute. Use the
  real `aiosqlite`; these functions take its rows.
* **`Row` and `row_factory`.** `column` and `column_stats` accept a row factory's
  output, indexing by position or by column name, but they do not implement one.
* **Text and BLOB *search*.** `LIKE`, `instr` and the other predicates are
  SQLite's to evaluate. `select_rows` takes a boolean mask the caller has
  already computed; the kernel copies the rows it names.
* **`groupby`, `ORDER BY` and any query planning.** SQLite's.
* **NaN in a numeric column.** The min and max are comparisons, and neither this
  nor SQL defines a total order over NaN. Not covered and not tested.
* **`blob_offsets` is the source offsets only.** There is no destination offset
  table for a partially selected batch exposed on its own; `pack_blobs` computes
  it internally.

## Install

```bash
bash build/build.sh          # -> dist/libmojo-aiosqlite.so
PYTHONPATH=python python -m pytest tests -q
PYTHONPATH=python python bench/bench.py
```

The repository pins its own Mojo toolchain in `pixi.toml`
(`mojo == 1.2.0.dev2026092605`); use the shared environment rather than
`pixi install`. Set `PYTHONPATH=python` when using the package outside a task.

`pytest-asyncio` is not installed in the shared test environment, so
`tests/conftest.py` runs coroutine tests with `asyncio.run` through a
`pytest_pyfunc_call` hook.

## Performance

Best-of-N wall clock, same process. Every case checks its result *before* it is
timed - the BLOB cases against the bytes they must reproduce, the statistics
against SQL's own aggregates run through real `aiosqlite`.

These numbers come from a shared build box that was running ~450 other runnable
tasks at the time (load average 447), so treat them as an order of magnitude
rather than three significant figures. Run-to-run variance on the reference
columns was 2-8x; the ranking of the cases was stable across five runs.

| case | python | mojo-aiosqlite | result |
| --- | ---: | ---: | ---: |
| BLOB offsets, 131072 rows | 17.94 ms | 0.28 ms | 64.53x faster |
| pack_blobs, 16384 rows, 4.2 MB | 28.05 ms | 41.18 ms | **0.68x - slower** |
| column_stats, 1M rows, end to end | 254.39 ms | 506.60 ms | **0.50x - slower** |
| ... the same kernel, column prebuilt | - | 9.98 ms | NumPy: 32.14 ms |

Two of those are losses and both are worth explaining.

**`pack_blobs` is slower, and it should be.** A byte gather is bandwidth-bound
and already at memory speed in Python: `out += flat[start:start + n]` is a
`memcpy` behind a slice object. The Mojo kernel copies the same 4.2 MB in
20.02 ms on prebuilt buffers - about 200 MB/s, which is byte-at-a-time
through bounds-checked pointer indexing rather than a vectorised block copy. The
compilation brief is explicit that memory-bound kernels are not where a
compiled inner loop helps, and this is a live demonstration of it. The kernel is
here because the offsets and the selection it consumes are the same arithmetic
the other two cases need, not because it beats a slice.

**`column_stats` is slower end to end, for the same reason at a larger scale.**
Getting a column out of 1M `sqlite3.Row` objects is a Python-level pass no
matter what happens afterwards, and it costs about 500 ms. The kernel itself is
fast: on the prebuilt column it takes 9.98 ms, against 32.14 ms for the
equivalent NumPy pass and 254.39 ms for the hand-written Python loop - so the
compiled arithmetic is 25x faster than the loop it replaces. The API is still
net slower because the extraction that precedes it costs more than the saving.
That is why `column`, `stats_f64` and `stats_i64` are all public: a caller who
already has the column as an array, or who can read it once, should call the
kernel directly and skip the extraction.

**The offsets scan is the real win**, and it is a real one: 0.28 ms against
17.94 ms for the Python loop and 1.59 ms for `np.cumsum`. A prefix sum has no
temporaries and no library call overhead, so a single compiled pass wins.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit. `build/build.sh`
compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-aiosqlite.so`.

`python/mojo_aiosqlite` owns every array. Buffers cross the C ABI as 64-bit
addresses (`ctypes.c_int64`; `c_int` truncates and segfaults) and are rebuilt
inside the kernel as `Pointer[Int64, ...]`, `Pointer[Float64, ...>` or
`Pointer[UInt8, ...]`, which keeps the exported symbols non-parametric.

**The kernels are left serial.** A prefix sum, a gather and a single-pass
reduction are all one streaming traversal with about one operation per element,
far below the ~2 flops/byte where chunking a thread pool pays, and the stats
kernel is a dependent accumulation besides. No `parallelize` is used: 1.2.0
cannot pass pointers into a parallel body.

The destination buffer of `pack_blobs` is sized by the sum of the *selected*
payload lengths, not by the last offset. `dst_offsets` is an exclusive prefix
sum, so a trailing empty payload would otherwise leave the buffer a slot short
of where the kernel writes - which is not a wrong answer, it is memory
corruption. The tests include empty payloads at the end of a column for that
reason.

## Numerical parity

| operation | agreement | asserted |
| --- | --- | --- |
| BLOB offsets, scan, gather | index arithmetic and byte copies | exact: `assert_array_equal`, `==` on bytes |
| count, NULL count, min, max | selections and integer counters | exact, and compared against SQL |
| sum | a running float64 total, in the same order as SQL's | `rel=1e-12` against SQL |

`stats_i64` accumulates the sum in `float64` because that is what SQLite does
once a running total leaves the 64-bit range; the count, the NULL count, the
minimum and the maximum are exact. Integer columns themselves stay `int64`
end to end, which a test checks against values at 2**62.

## Tests

`tests/test_rows.py`, 18 tests. Rows come from a real `aiosqlite` connection
against a real in-memory SQLite database, and the numeric reference is
**SQLite itself**: `SELECT count(x), sum(x), min(x), max(x)` over the same rows
through the same connection. A kernel that disagreed with SQL about NULLs would
fail there. Covered: REAL and INTEGER columns, NULLs in a third of the rows, a
column that is entirely NULL against SQL's `None` aggregates, a single row,
signed values, an empty batch, addressing by column name via `aiosqlite.Row`,
integer exactness at 2**62, the offsets table locating every payload including
empty ones, the full-column gather byte-for-byte, a `keep` mask, a selection
with `-1` holes, and out-of-range and length-mismatch errors.

"""Correctness-gated benchmark for mojo-aiosqlite.

Every case checks its result before it is timed - the BLOB cases against the
bytes they are supposed to reproduce, the statistics against SQL's own
aggregates run through real aiosqlite. The baselines are the Python a caller
actually writes: a prefix sum built with a loop, a loop of buffer slices for the
gather, and a single hand-written pass for the statistics.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import aiosqlite  # noqa: E402

import mojo_aiosqlite as mas  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


async def _run(coro_fn, *args):
    return await coro_fn(*args)


def _run_sync(coro_fn, *args):
    return asyncio.run(coro_fn(*args))


async def _fetch(values, tallies, blobs):
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE t (name TEXT, v REAL, k INTEGER, p BLOB)"
    )
    await conn.executemany(
        "INSERT INTO t VALUES (?, ?, ?, ?)",
        [(f"n{i:07d}", v, k, b) for i, (v, k, b) in enumerate(zip(values, tallies, blobs))],
    )
    rows = await (await conn.execute("SELECT name, v, k, p FROM t")).fetchall()
    return conn, rows


def bench_offsets(n_rows: int = 1 << 17, mean_len: int = 48):
    """The offsets table of a BLOB column: a prefix sum over the payload lengths."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(mean_len // 2, mean_len * 2, n_rows).astype(np.int64)

    want, total = mas.scan_offsets(lengths)
    running = 0
    ref_list = []
    for length in lengths.tolist():
        ref_list.append(running)
        running += int(length)
    np.testing.assert_array_equal(want, np.asarray(ref_list))
    assert total == running

    ref = _time(lambda: [None] and _python_scan(lengths.tolist()), 3)
    mine = _time(lambda: mas.scan_offsets(lengths), 5)
    numpy_time = _time(lambda: np.concatenate(([0], np.cumsum(lengths)[:-1])), 5)
    print(f"    (same lengths through np.cumsum: {numpy_time * 1e3:.2f} ms)")
    return f"blob offsets {n_rows} rows", ref, mine


def _python_scan(lengths):
    out = []
    running = 0
    for length in lengths:
        out.append(running)
        running += length
    return out


def bench_pack_blobs(n_rows: int = 1 << 14, mean_len: int = 256):
    """The gather: copy the selected payloads into one compact buffer."""
    rng = np.random.default_rng(1)
    blobs = [rng.bytes(int(rng.integers(1, mean_len * 2))) for _ in range(n_rows)]
    conn, rows = _run_sync(_fetch, [0.0] * n_rows, [0] * n_rows, blobs)
    lengths = np.array([len(row[3]) for row in rows], dtype=np.int64)
    src_offsets, _ = mas.scan_offsets(lengths)
    flat = b"".join(row[3] for row in rows)
    selection = np.arange(n_rows, dtype=np.int64)
    selected_lengths = lengths
    dst_offsets, _ = mas.scan_offsets(selected_lengths)

    got = mas.pack_blobs(rows, 3)
    assert got == flat, "packed blobs differ from the joined column"

    def python_gather():
        out = bytearray()
        for i in range(n_rows):
            start = int(src_offsets[i])
            out += flat[start:start + int(lengths[i])]
        return bytes(out)

    assert python_gather() == flat

    ref = _time(python_gather, 3)
    mine = _time(lambda: mas.pack_blobs(rows, 3), 3)
    kernel = _time(
        lambda: mas._lib.pack_blobs(flat, src_offsets, lengths, selection, dst_offsets), 3
    )
    print(f"    (kernel alone on the prebuilt buffers: {kernel * 1e3:.2f} ms)")
    print(f"    ({sum(lengths.tolist()) / 1e6:.1f} MB of payload)")
    return f"pack_blobs {n_rows} rows", ref, mine


def bench_column_stats(n_rows: int = 1 << 20):
    """The fused statistics pass, against a hand-written Python pass."""
    rng = np.random.default_rng(2)
    values = rng.standard_normal(n_rows).tolist()
    for i in range(0, n_rows, 7):
        values[i] = None
    conn, rows = _run_sync(_fetch, values, [0] * n_rows, [b""] * n_rows)

    stats = mas.column_stats(rows, 1)
    async def sql_totals():
        async with conn.execute(
            "SELECT count(v), sum(v), min(v), max(v) FROM t"
        ) as cur:
            return await cur.fetchone()

    c, s, lo, hi = _run_sync(sql_totals)
    assert stats.count == c
    assert stats.total == s or abs(stats.total - s) <= 1e-9 * max(abs(s), 1.0)
    assert stats.minimum == lo and stats.maximum == hi

    raw = [row[1] for row in rows]

    def python_stats():
        seen = nulls = 0
        total = 0.0
        lo = hi = None
        for v in raw:
            if v is None:
                nulls += 1
                continue
            if seen == 0:
                lo = hi = v
            else:
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
            total += v
            seen += 1
        return seen, nulls, total, lo, hi

    assert python_stats() == (stats.count, stats.nulls, stats.total, stats.minimum, stats.maximum)

    values_arr, isnull_arr = mas.column(rows, 1)

    ref = _time(python_stats, 3)
    mine = _time(lambda: mas.column_stats(rows, 1), 3)
    kernel = _time(lambda: mas._lib.stats_f64(values_arr, isnull_arr), 5)
    numpy_time = _time(
        lambda: (
            int((~isnull_arr.astype(bool)).sum()),
            float(values_arr[~isnull_arr.astype(bool)].sum()),
            float(values_arr.min()),
            float(values_arr.max()),
        ),
        5,
    )
    print(f"    (kernel alone on the prebuilt column: {kernel * 1e3:.2f} ms)")
    print(f"    (same column through NumPy: {numpy_time * 1e3:.2f} ms)")
    return f"column_stats {n_rows} rows", ref, mine


def main():
    print(f"{'case':<32}{'python':>12}{'mojo-aiosqlite':>18}{'ratio':>10}")
    print(f"{'':<32}{'':>12}{'':>18}{'(ref/mojo)':>10}")
    print("-" * 72)
    for fn in (bench_offsets, bench_pack_blobs, bench_column_stats):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        verdict = "faster" if ratio > 1.02 else ("slower" if ratio < 0.98 else "parity")
        print(f"{label:<32}{ref * 1e3:>10.2f}ms{got * 1e3:>16.2f}ms{ratio:>9.2f}x  {verdict}")


if __name__ == "__main__":
    main()

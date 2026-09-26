"""Parity against real aiosqlite fetches and against SQL's own aggregates.

The rows come from a real `aiosqlite` connection. The numeric reference is not
Python - it is SQLite itself: `SELECT count(x), sum(x), min(x), max(x)` run over
the same rows through the same connection, so a kernel that disagrees with SQL's
aggregate semantics about NULLs fails here.

The BLOB and offset results are byte and index arithmetic, so they are asserted
with `==` and `assert_array_equal`. The only member of `ColumnStats` that is not
bit-exact is the sum, which is a running float64 total, and that one is compared
against SQL with an explicit tolerance.
"""

import math

import aiosqlite
import numpy as np
import pytest

import mojo_aiosqlite as mas

SUM_RTOL = 1e-12

SCHEMA = """
CREATE TABLE samples (
    id INTEGER PRIMARY KEY,
    name TEXT,
    value REAL,
    tally INTEGER,
    payload BLOB
);
"""


async def fresh_db(rows):
    """An in-memory database populated through real aiosqlite."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(SCHEMA)
    await conn.executemany(
        "INSERT INTO samples (name, value, tally, payload) VALUES (?, ?, ?, ?)", rows
    )
    await conn.commit()
    return conn


async def ordered_rows(conn, columns: str, key: str = "name"):
    """The table's rows in insertion order, selecting the named columns."""
    cursor = await conn.execute(f"SELECT {columns} FROM samples ORDER BY {key}")
    rows = await cursor.fetchall()
    await cursor.close()
    return rows


def values(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).tolist()


def tallies(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 1000, n).tolist()


def payloads(n, seed=2):
    rng = np.random.default_rng(seed)
    return [rng.bytes(rng.integers(0, 64)) for _ in range(n)]


def blob_rows(blobs):
    return [(f"n{i:04d}", None, None, blob) for i, blob in enumerate(blobs)]


# --------------------------------------------------------------------------
# numeric columns, against SQL's own aggregates


async def test_column_stats_match_sql_for_a_real_column():
    n = 500
    rows = [
        (f"n{i:04d}", v, c, b"")
        for i, (v, c) in enumerate(zip(values(n), tallies(n)))
    ]
    conn = await fresh_db(rows)
    ordered = await ordered_rows(conn, "value, tally")

    stats = mas.column_stats(ordered, 0)
    async with conn.execute(
        "SELECT count(value), sum(value), min(value), max(value) FROM samples"
    ) as cur:
        c, s, lo, hi = await cur.fetchone()
    assert stats.count == c == n
    assert stats.nulls == 0
    assert stats.total == pytest.approx(s, rel=SUM_RTOL)
    assert stats.minimum == lo
    assert stats.maximum == hi
    await conn.close()


async def test_column_stats_exclude_nulls_like_sql_does():
    n = 300
    vals = values(n)
    rows = [
        (f"n{i:04d}", (v if i % 3 else None), c, b"")
        for i, (v, c) in enumerate(zip(vals, tallies(n)))
    ]
    conn = await fresh_db(rows)
    ordered = await ordered_rows(conn, "value, tally")

    stats = mas.column_stats(ordered, 0)
    async with conn.execute(
        "SELECT count(value), sum(value), min(value), max(value) FROM samples"
    ) as cur:
        c, s, lo, hi = await cur.fetchone()
    assert stats.count == c == n - len(vals[::3])
    assert stats.nulls == n - c
    assert stats.total == pytest.approx(s, rel=SUM_RTOL)
    assert stats.minimum == lo
    assert stats.maximum == hi
    assert mas.null_count(ordered, 0) == stats.nulls
    assert mas.null_count(ordered, 1) == 0
    await conn.close()


async def test_integer_column_stats_match_sql():
    n = 400
    rows = [(f"n{i:04d}", None, c, b"") for i, c in enumerate(tallies(n))]
    conn = await fresh_db(rows)
    ordered = await ordered_rows(conn, "value, tally")

    stats = mas.column_stats(ordered, 1)
    async with conn.execute(
        "SELECT count(tally), sum(tally), min(tally), max(tally) FROM samples"
    ) as cur:
        c, s, lo, hi = await cur.fetchone()
    assert stats.count == c == n
    assert stats.total == pytest.approx(s, rel=SUM_RTOL)
    assert stats.minimum == lo
    assert stats.maximum == hi
    await conn.close()


async def test_a_column_of_only_nulls_reports_sql_null_aggregates():
    n = 25
    rows = [(f"n{i:04d}", None, None, b"") for i in range(n)]
    conn = await fresh_db(rows)
    ordered = await ordered_rows(conn, "value, tally")

    for index in (0, 1):
        stats = mas.column_stats(ordered, index)
        assert stats.count == 0
        assert stats.nulls == n
        assert math.isnan(stats.total)
        assert math.isnan(stats.minimum)
        assert math.isnan(stats.maximum)
    async with conn.execute("SELECT sum(value), min(value), max(value) FROM samples") as cur:
        s, lo, hi = await cur.fetchone()
    assert s is None and lo is None and hi is None
    await conn.close()


async def test_single_row_and_signed_columns():
    conn = await fresh_db([("only", -17.5, -9, b"")])
    ordered = await ordered_rows(conn, "value, tally")
    stats = mas.column_stats(ordered, 0)
    assert (stats.count, stats.nulls, stats.total) == (1, 0, -17.5)
    assert (stats.minimum, stats.maximum) == (-17.5, -17.5)
    ints = mas.column_stats(ordered, 1)
    assert (ints.count, ints.total, ints.minimum, ints.maximum) == (1, -9.0, -9.0, -9.0)
    await conn.close()


async def test_column_stats_on_an_empty_batch():
    stats = mas.column_stats([], 0)
    assert stats.count == 0 and stats.nulls == 0
    assert math.isnan(stats.total)
    assert mas.null_count([], 0) == 0


async def test_columns_can_be_addressed_by_name():
    conn = await fresh_db([("a", 1.5, 2, b"x"), ("b", 2.5, 3, b"yy")])
    cursor = await conn.execute("SELECT * FROM samples ORDER BY id")
    cursor.row_factory = aiosqlite.Row
    ordered = await cursor.fetchall()
    assert [r["name"] for r in ordered] == ["a", "b"]
    assert mas.column_stats(ordered, "value").total == pytest.approx(4.0, rel=SUM_RTOL)
    assert mas.column_stats(ordered, "tally").count == 2
    # `SELECT *` puts the TEXT name in column 1, so this is a negative test.
    with pytest.raises(ValueError):
        mas.column_stats(ordered, 1)
    await conn.close()


async def test_column_materialisation_keeps_integers_exact():
    big = [2**62, -(2**62), 0]
    conn = await fresh_db([(f"n{i}", None, v, b"") for i, v in enumerate(big)])
    ordered = await ordered_rows(conn, "tally")
    vals, isnull = mas.column(ordered, 0)
    assert vals.dtype == np.int64
    np.testing.assert_array_equal(vals, np.array(big, dtype=np.int64))
    np.testing.assert_array_equal(isnull, np.zeros(3, dtype=np.uint8))
    await conn.close()


# --------------------------------------------------------------------------
# BLOB columns


async def test_blob_offsets_locate_every_payload():
    blobs = payloads(64)
    conn = await fresh_db(blob_rows(blobs))
    ordered = await ordered_rows(conn, "payload")

    offsets, total = mas.blob_offsets(ordered, 0)
    assert total == sum(len(b) for b in blobs)
    assert offsets.size == len(blobs)

    # The offsets address the concatenated buffer, so check them against that,
    # not against the individual payload objects.
    flat = b"".join(row[0] for row in ordered)
    assert len(flat) == total
    for i, blob in enumerate(blobs):
        start = int(offsets[i])
        assert flat[start:start + len(blob)] == blob
    await conn.close()


async def test_pack_blobs_reproduces_the_joined_column_byte_for_byte():
    blobs = payloads(200)
    conn = await fresh_db(blob_rows(blobs))
    ordered = await ordered_rows(conn, "payload")
    assert mas.pack_blobs(ordered, 0) == b"".join(blobs)
    await conn.close()


async def test_pack_blobs_gathers_only_the_selected_rows():
    blobs = payloads(128)
    conn = await fresh_db(blob_rows(blobs))
    ordered = await ordered_rows(conn, "payload")

    keep = np.array([i % 3 == 0 for i in range(len(blobs))])
    selected = mas.select_rows(ordered, 0, keep)
    packed = mas.pack_blobs(ordered, 0, selected)
    assert packed == b"".join(blobs[i] for i in range(len(blobs)) if i % 3 == 0)
    await conn.close()


async def test_selection_may_contain_holes():
    blobs = payloads(16)
    conn = await fresh_db(blob_rows(blobs))
    ordered = await ordered_rows(conn, "payload")

    selected = np.array(
        [0, -1, 2, -1, -1, 5, -1, 7, -1, 9, -1, 11, -1, -1, 14, -1], dtype=np.int64
    )
    packed = mas.pack_blobs(ordered, 0, selected)
    assert packed == (
        blobs[0] + blobs[2] + blobs[5] + blobs[7] + blobs[9] + blobs[11] + blobs[14]
    )
    await conn.close()


async def test_blob_roundtrip_preserves_content_not_just_length():
    """Length-only equality would pass; the bytes have to come back right."""
    blobs = [bytes([i % 256]) * (i % 37 + 1) for i in range(64)]
    conn = await fresh_db(blob_rows(blobs))
    ordered = await ordered_rows(conn, "payload")
    packed = mas.pack_blobs(ordered, 0)
    offsets, _ = mas.blob_offsets(ordered, 0)
    for i, blob in enumerate(blobs):
        start = int(offsets[i])
        assert packed[start:start + len(blob)] == blob
    await conn.close()


async def test_empty_blobs_and_selection_errors():
    conn = await fresh_db(
        [("a", None, None, b""), ("b", None, None, b"abc"), ("c", None, None, b"")]
    )
    ordered = await ordered_rows(conn, "payload")
    offsets, total = mas.blob_offsets(ordered, 0)
    assert total == 3
    np.testing.assert_array_equal(offsets, [0, 0, 3])
    assert mas.pack_blobs(ordered, 0) == b"abc"
    assert mas.pack_blobs(ordered, 0, np.array([1, -1, 0], dtype=np.int64)) == b"abc"

    with pytest.raises(ValueError):
        mas.select_rows(ordered, 0, np.array([True]))
    with pytest.raises(IndexError):
        mas.pack_blobs(ordered, 0, np.array([99], dtype=np.int64))
    await conn.close()


async def test_blob_offsets_of_an_empty_batch():
    offsets, total = mas.blob_offsets([], 0)
    assert offsets.size == 0 and total == 0
    assert mas.pack_blobs([], 0) == b""


def test_scan_offsets_matches_numpy():
    lengths = np.array([0, 3, 0, 5, 0, 7, 0], dtype=np.int64)
    offsets, total = mas.scan_offsets(lengths)
    np.testing.assert_array_equal(offsets, np.concatenate(([0], np.cumsum(lengths)[:-1])))
    assert total == int(lengths.sum())
    with pytest.raises(ValueError):
        mas.scan_offsets([-1, 2])


# --------------------------------------------------------------------------
# the real aiosqlite is untouched and still works


async def test_aiosqlite_still_imports_and_works_alongside():
    assert aiosqlite.__name__ == "aiosqlite"
    assert mas.__name__ == "mojo_aiosqlite"
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE TABLE t (x INTEGER)")
    await conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(10)])
    rows = await (await conn.execute("SELECT x FROM t")).fetchall()
    assert len(rows) == 10
    assert mas.column_stats(rows, 0).count == 10
    await conn.close()


async def test_iter_chunked_batches_feed_the_same_helpers():
    """The batch this package targets is what `Cursor.__aiter__` yields."""
    n = 300
    rows = [
        (f"n{i:04d}", v, c, b"")
        for i, (v, c) in enumerate(zip(values(n), tallies(n)))
    ]
    conn = await fresh_db(rows)
    conn.iter_chunk_size = 64
    cursor = await conn.execute("SELECT value FROM samples")
    seen, total, count = 0, 0.0, 0
    async for row in cursor:
        seen += 1
        stats = mas.column_stats([row], 0)
        count += stats.count
        total += stats.total
    assert seen == n
    assert count == n
    async with conn.execute("SELECT sum(value) FROM samples") as cur:
        (s,) = await cur.fetchone()
    assert total == pytest.approx(s, rel=1e-10)
    await conn.close()

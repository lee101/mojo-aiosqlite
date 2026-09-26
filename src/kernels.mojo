"""Row-batch marshalling arithmetic for aiosqlite.

`aiosqlite` is 1483 lines, 537 of which are its test suite, and it is a thread
bridge: `Connection._execute` hands a callable to a worker thread and resolves a
future with the result. It performs no arithmetic of its own. What it does move
is *data*: a fetched batch of rows, whose columns are BLOBs and numbers.

Two things happen to such a batch that are real loops over real bytes, and
those are what is compiled here:

* the **offsets table** for a BLOB column - an exclusive prefix sum over the
  per-row payload lengths, which is what turns a list of `bytes` objects into
  one addressable buffer;
* the **gather** that copies the selected rows' payloads out of a flat source
  buffer into a compact destination, a scattered run of `memcpy`s;
* the **column statistics** of a numeric column, fused into one pass that
  produces the count of non-NULLs, the count of NULLs, the sum, the minimum and
  the maximum with SQL's aggregate semantics - NULLs excluded from every one of
  them.

All of it is Int64 index arithmetic or a byte copy, so the results are exact and
the tests assert equality rather than a tolerance.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime MPtr = Pointer[UInt8, AnyOrigin[mut=True]]


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


@export("asq_scan_offsets")
def asq_scan_offsets(
    lengths_addr: Int, offsets_addr: Int, n: Int, total_addr: Int
) abi("C"):
    """Exclusive prefix sum of the BLOB payload lengths.

    `offsets[i]` is where row `i`'s payload starts in the flat buffer, so row
    `i` occupies `[offsets[i], offsets[i] + lengths[i])`. The total payload size
    is written to `total_addr`.
    """
    var lengths = ip(lengths_addr)
    var offsets = ip(offsets_addr)
    var total = ip(total_addr)
    var running = Int64(0)
    for i in range(n):
        offsets[unsafe_offset=i] = running
        var length = lengths[unsafe_offset=i]
        if length > 0:
            running += length
    total[unsafe_offset=0] = running


@export("asq_pack_blobs")
def asq_pack_blobs(
    src_addr: Int,
    src_offsets_addr: Int,
    lengths_addr: Int,
    rows_addr: Int,
    dst_offsets_addr: Int,
    dst_addr: Int,
    n: Int,
    total_addr: Int,
) abi("C"):
    """Copy the payloads of the selected rows into one compact buffer.

    `src_offsets[i]` and `lengths[i]` locate row `i`'s payload in the flat source
    buffer. `rows[j]` is the source row to copy j-th, and `dst_offsets[j]` is
    where it lands in the destination. Rows are copied in the order given, so the
    destination offsets have to be the prefix sum of the *selected* lengths; the
    number of bytes written is returned through `total_addr` for the caller to
    check.

    A row index of -1 is skipped, which is how a selection mask with holes is
    expressed without compacting the index array first.
    """
    var src = bp(src_addr)
    var src_offsets = ip(src_offsets_addr)
    var lengths = ip(lengths_addr)
    var rows = ip(rows_addr)
    var dst_offsets = ip(dst_offsets_addr)
    var dst = bp(dst_addr)
    var total = ip(total_addr)
    var written = Int64(0)
    for j in range(n):
        var row = rows[unsafe_offset=j]
        if row < 0:
            continue
        var length = lengths[unsafe_offset=row]
        if length <= 0:
            continue
        var origin = src_offsets[unsafe_offset=row]
        var target = dst_offsets[unsafe_offset=j]
        var k = Int64(0)
        while k < length:
            dst[unsafe_offset=target + k] = src[unsafe_offset=origin + k]
            k += 1
        written += length
    total[unsafe_offset=0] = written


@export("asq_stats_f64")
def asq_stats_f64(
    values_addr: Int, isnull_addr: Int, out_addr: Int, n: Int
) abi("C"):
    """One-pass statistics of a REAL column, with SQL aggregate semantics.

    Writes `[count, nulls, sum, min, max]`, where `count` excludes NULLs and
    `nulls` counts them. The extrema of a NULL-only column are left at zero and
    the caller substitutes NaN, which is what SQL's NULL aggregates become in
    Python.

    `values` holds the non-NULL values; the entries for NULL rows are ignored,
    so they may be left as zero. `isnull` is a byte mask.
    """
    var values = fp(values_addr)
    var isnull = bp(isnull_addr)
    var out = fp(out_addr)
    var seen = Int64(0)
    var nulls = Int64(0)
    var total = 0.0
    var lo = 0.0
    var hi = 0.0
    for i in range(n):
        if isnull[unsafe_offset=i] != 0:
            nulls += 1
            continue
        var v = values[unsafe_offset=i]
        if seen == 0:
            lo = v
            hi = v
        else:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        total += v
        seen += 1
    out[unsafe_offset=0] = Float64(seen)
    out[unsafe_offset=1] = Float64(nulls)
    out[unsafe_offset=2] = total
    # A NULL-only column leaves the extrema at zero; the caller knows the count
    # and substitutes NaN, which is what SQL's NULL aggregates become in Python.
    out[unsafe_offset=3] = lo
    out[unsafe_offset=4] = hi


@export("asq_stats_i64")
def asq_stats_i64(
    values_addr: Int, isnull_addr: Int, out_addr: Int, n: Int
) abi("C"):
    """One-pass statistics of an INTEGER column, with SQL aggregate semantics.

    Same five outputs as `asq_stats_f64`. The sum accumulates in `float64`
    because that is what SQLite does once a running total leaves the 64-bit
    integer range; the count and the null count are exact integers, and so are
    the minimum and the maximum while the column holds only 64-bit values.
    """
    var values = ip(values_addr)
    var isnull = bp(isnull_addr)
    var out = fp(out_addr)
    var seen = Int64(0)
    var nulls = Int64(0)
    var total = 0.0
    var lo = 0.0
    var hi = 0.0
    for i in range(n):
        if isnull[unsafe_offset=i] != 0:
            nulls += 1
            continue
        var raw = values[unsafe_offset=i]
        var v = Float64(raw)
        if seen == 0:
            lo = v
            hi = v
        else:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        total += v
        seen += 1
    out[unsafe_offset=0] = Float64(seen)
    out[unsafe_offset=1] = Float64(nulls)
    out[unsafe_offset=2] = total
    # A NULL-only column leaves the extrema at zero; the caller knows the count
    # and substitutes NaN, which is what SQL's NULL aggregates become in Python.
    out[unsafe_offset=3] = lo
    out[unsafe_offset=4] = hi


@export("asq_null_count")
def asq_null_count(isnull_addr: Int, out_addr: Int, n: Int) abi("C"):
    """Count the set bytes of a NULL mask. Exact, and the cheapest of the lot."""
    var isnull = bp(isnull_addr)
    var out = ip(out_addr)
    var nulls = Int64(0)
    for i in range(n):
        if isnull[unsafe_offset=i] != 0:
            nulls += 1
    out[unsafe_offset=0] = nulls

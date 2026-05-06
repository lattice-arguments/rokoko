#!/usr/bin/env python3
"""Diff two rokoko snapshot JSONs.

Loads both snapshots, prints a markdown table of per-span time deltas
sorted by absolute change descending. Warns on metadata mismatches
(different features or machine), but the comparison still runs.

Usage:
    python3 bench_results/diff_snapshots.py <baseline.json> <candidate.json>
"""
import json
import sys

NANOS_PER_SECOND = 1_000_000_000


def fmt_ns(ns):
    """Render a nanosecond count with sensible units."""
    if ns is None:
        return "—"
    abs_ns = abs(ns)
    if abs_ns >= NANOS_PER_SECOND:
        return f"{ns / NANOS_PER_SECOND:.2f} s"
    if abs_ns >= 1_000_000:
        return f"{ns / 1_000_000:.0f} ms"
    if abs_ns >= 1_000:
        return f"{ns / 1_000:.0f} μs"
    return f"{ns} ns"


def fmt_signed_ns(ns):
    if ns is None:
        return "—"
    sign = "+" if ns > 0 else ""
    return sign + fmt_ns(ns)


def main():
    if len(sys.argv) != 3:
        print(
            "usage: diff_snapshots.py <baseline.json> <candidate.json>",
            file=sys.stderr,
        )
        sys.exit(2)

    with open(sys.argv[1]) as f:
        baseline = json.load(f)
    with open(sys.argv[2]) as f:
        candidate = json.load(f)

    base_meta = baseline.get("metadata", {})
    cand_meta = candidate.get("metadata", {})

    print(
        f"# baseline `{base_meta.get('git_sha', '?')}` "
        f"→ candidate `{cand_meta.get('git_sha', '?')}`"
    )
    print()

    # A git_sha mismatch is expected (the whole point of diffing). Features
    # and machine mismatches are noteworthy because they change what the
    # timings mean — flag them but still produce the table.
    warnings = []
    for key in ("features", "machine"):
        if base_meta.get(key) != cand_meta.get(key):
            warnings.append(
                f"{key} mismatch: baseline=`{base_meta.get(key)}`, "
                f"candidate=`{cand_meta.get(key)}`"
            )
    if warnings:
        print("⚠️ Metadata mismatches:")
        for w in warnings:
            print(f"- {w}")
        print()

    base_spans = baseline.get("spans", {})
    cand_spans = candidate.get("spans", {})
    all_names = sorted(set(base_spans) | set(cand_spans))

    rows = []
    for name in all_names:
        b = base_spans.get(name, {})
        c = cand_spans.get(name, {})
        b_ns = b.get("total_ns")
        c_ns = c.get("total_ns")
        calls = c.get("calls", b.get("calls", 0))
        if b_ns is not None and c_ns is not None:
            delta_ns = c_ns - b_ns
            delta_pct = (delta_ns / b_ns) * 100 if b_ns > 0 else None
        elif c_ns is not None:
            delta_ns = c_ns  # span new on candidate side
            delta_pct = None
        else:
            delta_ns = -(b_ns or 0)  # span removed on candidate side
            delta_pct = None
        rows.append((name, b_ns, c_ns, calls, delta_ns, delta_pct))

    rows.sort(key=lambda r: -abs(r[4] or 0))

    print("| span | baseline | candidate | calls | Δ | Δ% |")
    print("|------|---------:|----------:|:-----:|---:|---:|")
    for name, b_ns, c_ns, calls, delta_ns, delta_pct in rows:
        b_str = fmt_ns(b_ns)
        c_str = fmt_ns(c_ns)
        d_str = fmt_signed_ns(delta_ns)
        if delta_pct is None:
            p_str = "—"
        else:
            sign = "+" if delta_pct >= 0 else ""
            p_str = f"{sign}{delta_pct:.1f}%"
        print(f"| `{name}` | {b_str} | {c_str} | {calls} | {d_str} | {p_str} |")


if __name__ == "__main__":
    main()

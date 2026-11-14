#!/usr/bin/env python3
"""Summarize all samples, without removing slow calls or GC calls."""
import csv
import collections
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
groups = collections.defaultdict(list)
with (root / "samples.csv").open() as stream:
    for row in csv.DictReader(stream):
        groups[row["family"], row["m"], row["n"], row["width"], row["mode"]].append(row)
rows = []
for (family, m, n, width, mode), values in groups.items():
    times = [float(v["total_s"]) for v in values]
    row = dict(family=family, m=m, n=n, width=width, mode=mode, samples=len(values),
               median_ms=statistics.median(times)*1000,
               min_ms=min(times)*1000, max_ms=max(times)*1000,
               median_bytes=statistics.median(float(v["bytes"]) for v in values))
    for stage in "abc":
        row[f"stage_{stage}_ms"] = statistics.median(float(v[f"stage_{stage}_s"]) for v in values)*1000
    rows.append(row)
bases = {(r["family"], r["m"], r["n"], r["mode"]): r["median_ms"]
         for r in rows if r["width"] == "1"}
for row in rows:
    base = bases.get((row["family"], row["m"], row["n"], row["mode"]))
    row["speedup_vs_1t"] = base / row["median_ms"] if base and row["median_ms"] else ""
with (root / "summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(f'{row["family"]:17s} {row["m"]}x{row["n"]} {row["mode"]:6s} '
          f'{row["width"]:>2s}T {row["median_ms"]:9.3f} ms '
          f'{row["median_bytes"]:10.0f} bytes speedup={row["speedup_vs_1t"]}')

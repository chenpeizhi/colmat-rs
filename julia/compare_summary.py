#!/usr/bin/env python3
"""Compare isolated backend process medians within matched starts; retain all guards."""
import collections
import csv
import json
import pathlib
import statistics as st
import sys
root=pathlib.Path(sys.argv[1])
all_samples=[]
process=[]
hashes=collections.defaultdict(set)
for file in sorted(root.glob('s*-t*/samples.csv')):
    run=file.parent.name
    guard=json.loads((root/(run+'.guard.json')).read_text())
    rows=list(csv.DictReader(file.open()))
    assert len({r['backend'] for r in rows})==1,'use frozen mixed-scan summarizer for mixed processes'
    groups=collections.defaultdict(list)
    for r in rows:
        all_samples.append(dict(run=run,**r))
        key=r['family'],int(r['m']),int(r['n']),int(r['width']),r['backend']
        groups[key].append(float(r['total_s'])*1000)
        hashes[key[:3]].add(r['input_sha256'])
    for (family,m,n,width,backend),times in groups.items():
        process.append(dict(run=run,start=run.split('-')[0],family=family,m=m,n=n,width=width,
                            backend=backend,median_ms=st.median(times),
                            sample_min_ms=min(times),sample_max_ms=max(times),
                            idle_ok=guard['before']['all_below_5pct'] and guard['after']['all_below_5pct'],
                            affinity_ok=guard['affinity_ok']))
assert process and all(len(v)==1 for v in hashes.values()),'missing results or different inputs'
cohorts=collections.defaultdict(dict)
for r in process:
    cohorts[r['start'],r['family'],r['m'],r['n'],r['width']][r['backend']]=r
for cohort in cohorts.values():
    assert set(cohort)=={'colmat','julia_lab','julia_stdlib'},'incomplete cohort'
    for r in cohort.values():
        r['ratio_vs_colmat']=r['median_ms']/cohort['colmat']['median_ms']
groups=collections.defaultdict(list)
for r in process:
    groups[r['family'],r['m'],r['n'],r['width'],r['backend']].append(r)
summary=[]
for (family,m,n,width,backend),runs in sorted(groups.items()):
    times=[r['median_ms'] for r in runs]; ratios=[r['ratio_vs_colmat'] for r in runs]
    summary.append(dict(family=family,m=m,n=n,width=width,backend=backend,starts=len(runs),
                        median_ms=st.median(times),min_ms=min(times),max_ms=max(times),
                        ratio_vs_colmat=st.median(ratios),ratio_min=min(ratios),ratio_max=max(ratios),
                        idle_passed=sum(r['idle_ok'] for r in runs),affinity_passed=sum(r['affinity_ok'] for r in runs)))
for name,rows in [('all-samples.csv',all_samples),('paired-process.csv',process),('summary.csv',summary)]:
    with (root/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
for r in summary:
    print(f'{r["family"]:5s} {r["m"]}x{r["n"]} {r["width"]:2d}T {r["backend"]:12s} '
          f'{r["median_ms"]:9.3f}ms ratio={r["ratio_vs_colmat"]:.3f} '
          f'idle={r["idle_passed"]}/{r["starts"]}')

#!/usr/bin/env python3
"""Linux launcher: CPU IDs 1..16, shared host lock, retain endpoint load observations."""
import argparse
import fcntl
import json
import os
import pathlib
import subprocess
import time


def cpu_times():
    result = {}
    for line in pathlib.Path('/proc/stat').read_text().splitlines():
        key, *fields = line.split()
        if key.startswith('cpu') and key[3:].isdigit():
            values = list(map(int, fields[:8]))
            result[int(key[3:])] = (sum(values), values[3] + values[4])
    return result


def observe():
    before = cpu_times()
    time.sleep(3)
    after = cpu_times()
    # Include measured physical cores' SMT siblings. CPU17/35 are excluded.
    measured_domain = list(range(1, 17)) + list(range(19, 35))
    busy = {}
    for cpu in measured_domain:
        total = after[cpu][0] - before[cpu][0]
        idle = after[cpu][1] - before[cpu][1]
        busy[cpu] = (total-idle)/total if total else 0.0
    return {'busy_fraction': busy, 'all_below_5pct': all(v <= .05 for v in busy.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--lock', type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2] / 'colmat-benchmark/target/campaign.lock')
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('--threads must be in 1..16')
    if args.output.exists():
        parser.error('output directory already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The observer never runs on CPU17 or its sibling35. This affects this launcher only.
    os.sched_setaffinity(0, {0})
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open('a') as lock:
        print('Waiting for host lock:', args.lock, flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print('Lock acquired; observing load, then recording once.', flush=True)
        before = observe()
        cmd = ['taskset', '-c', '1-16', 'julia', '--project='+str(pathlib.Path(__file__).parent),
               '-t', f'{args.threads},0', str(pathlib.Path(__file__).parent/'measure.jl'), str(args.output)]
        started = time.time()
        process = subprocess.Popen(cmd)
        observed_masks = set()
        affinity_ok = True
        while process.poll() is None:
            time.sleep(.25)
            for status in pathlib.Path(f'/proc/{process.pid}/task').glob('*/status'):
                try:
                    line = next(x for x in status.read_text().splitlines() if x.startswith('Cpus_allowed_list:'))
                except (FileNotFoundError, ProcessLookupError):
                    continue
                mask = line.split(':', 1)[1].strip()
                observed_masks.add(mask)
                cpus = set()
                for part in mask.split(','):
                    ends = list(map(int, part.split('-')))
                    cpus.update(range(ends[0], ends[-1]+1))
                affinity_ok &= cpus <= set(range(1, 17))
        returncode = process.returncode
        after = observe()
        # Retain failed guards as reference observations; never retry until clean.
        args.output.mkdir(exist_ok=True)
        (args.output/'load-observations.json').write_text(json.dumps(dict(
            before=before, after=after, returncode=returncode, affinity_ok=affinity_ok, observed_thread_masks=sorted(observed_masks),
            wall_seconds=time.time()-started, command=cmd,
            note='Reference scan, no retries. One shared L3; no per-cell load guards or formal acceptance.'), indent=2)+'\n')
        if returncode:
            raise SystemExit(returncode)
        if not affinity_ok:
            raise SystemExit("a Julia thread escaped CPU IDs 1..16; results are invalid")
        print('Load guards:', before['all_below_5pct'], after['all_below_5pct'], flush=True)


if __name__ == '__main__':
    main()

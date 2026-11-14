#!/usr/bin/env python3
"""Paired colmat/Julia comparisons, fixed CPU IDs1..width (width <=16), no guard retries."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode = True
from run_pinned import observe

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent


def audit(pid, allowed, masks):
    good = True
    for status in pathlib.Path(f'/proc/{pid}/task').glob('*/status'):
        try:
            line = next(x for x in status.read_text().splitlines() if x.startswith('Cpus_allowed_list:'))
        except (FileNotFoundError, ProcessLookupError):
            continue
        mask = line.split(':', 1)[1].strip()
        masks.add(mask)
        actual = set()
        for part in mask.split(','):
            ends = list(map(int, part.split('-')))
            actual.update(range(ends[0], ends[-1]+1))
        good &= actual <= allowed
    return good


def fingerprint(paths):
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--widths', default='1,4,8,16')
    parser.add_argument('--sizes', default='256,512')
    parser.add_argument('--lock', type=pathlib.Path, default=REPO.parent/'colmat-benchmark/target/campaign.lock')
    args = parser.parse_args()
    widths = list(map(int, args.widths.split(',')))
    if args.starts < 1 or args.reps < 1 or not widths or any(w < 1 or w > 16 for w in widths):
        parser.error('positive starts/reps, widths in1..16 required')
    if args.output.exists():
        parser.error('output directory exists')
    library = REPO/'target/release/libcolmat_julia_compare.so'
    if not library.is_file():
        parser.error('build julia/colmat-bridge first (see README)')
    os.sched_setaffinity(0, {0})
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open('a') as lock:
        print('Waiting for host lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        args.output.mkdir(parents=True)
        files = sorted(p for p in HERE.rglob('*') if p.is_file()
                       and (p.suffix in ('.jl','.py','.toml','.rs','.md') or p.name=='Cargo.lock')
                       and '__pycache__' not in p.parts and 'target' not in p.parts)
        hashes = fingerprint(files)
        for p in files:
            dest = args.output/'source'/p.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        lib_hash = hashlib.sha256(library.read_bytes()).hexdigest()
        shutil.copy2(library,args.output/library.name)
        manifest = dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                        rust_source_tree=subprocess.check_output(['git','rev-parse','HEAD:faer/src'],cwd=REPO,text=True).strip(),
                        source_sha256=hashes, library_sha256=lib_hash,
                        starts=args.starts,reps=args.reps,widths=widths,sizes=args.sizes,
                        scope='f64, complete thin factors, allocation-inclusive owning calls, separate backend processes',
                        pairing='ratio of per-backend process medians within matched start/width/input',
                        note='One shared L3, fixed CPU IDs1..width, bounded spins settled20ms. Reference scan, no retries.')
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for start in range(args.starts):
            order = widths if start%2==0 else list(reversed(widths))
            for width in order:
                backends = ['colmat','julia_lab','julia_stdlib']
                shift = (start + widths.index(width)) % 3
                backends = backends[shift:] + backends[:shift]
                for backend in backends:
                    name = f's{start}-t{width}-{backend}'
                    dest = args.output/name
                    print(name,'CPU IDs',f'1-{width}',flush=True)
                    before = observe()
                    env = os.environ.copy()
                    env.update(SIZES=args.sizes,COLMAT_COMPARE_LIB=str(library),OPENBLAS_NUM_THREADS="1")
                    command = ['taskset','-c',f'1-{width}','julia','--project='+str(HERE),'-t',f'{width},0',
                               str(HERE/'compare.jl'),str(dest),str(width),str(args.reps),backend]
                    masks = set(); affinity_ok = True
                    with (args.output/(name+'.log')).open('w') as log:
                        process = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            time.sleep(.25)
                            affinity_ok &= audit(process.pid,set(range(1,width+1)),masks)
                    after = observe()
                    state = dict(before=before,after=after,affinity_ok=affinity_ok,
                                 observed_thread_masks=sorted(masks),returncode=process.returncode,command=command)
                    (args.output/(name+'.guard.json')).write_text(json.dumps(state,indent=2)+'\n')
                    print(name,'exit',process.returncode,'affinity',affinity_ok,
                          'idle',before['all_below_5pct'],after['all_below_5pct'],flush=True)
                    if process.returncode or not affinity_ok:
                        raise SystemExit(f'{name} failed; retained without retry, see log')
                    if fingerprint(files)!=hashes or hashlib.sha256(library.read_bytes()).hexdigest()!=lib_hash:
                        raise SystemExit('measured source or library changed; results invalid')
        print('Completed:',args.output,flush=True)


if __name__ == '__main__':
    main()

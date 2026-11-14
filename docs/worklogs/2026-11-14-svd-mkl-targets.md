# SVD target cases against MKL, and the 511/513 speedup (#83)

Issue: [colmat-rs#83](https://github.com/tensor4all/colmat-rs/issues/83).

## Target and evidence contract

The requested target is at least 1.1 times MKL's throughput (colmat elapsed time
at most MKL/1.1) at 1024x64, 128x128 and 256x256. The added 511x511 and 513x513
cases are measured alongside 512x512 to expose leading-dimension boundaries.
The active goal also includes QR at 511/513: prioritize at least 10% lower
elapsed time than the current implementation, with 5% accepted by the user
as the minimum. Report each target case separately in milliseconds.
Primary coverage is f64/c64, C/F layout, values-only/thin SVD, 1/4/8 threads.
The four native types remain numerical and non-regression guards.

Baseline colmat is `a395150c086f`; candidate `a407a0b6`. Harness
`f3cd501` (the local SVD diagnostic with the 511/513 grid and per-case target
columns), MKL 2025.3, Rust 1.96.0, release, Ryzen 8945HS. Both runs are in
sibling colmat-benchmark: baseline `data/probes/svd-docker/20261114T101544Z/`,
candidate `data/probes/svd-docker/20261114T115445Z/`. The host is the
`zen4-8945hs-dev` profile with boost on, one shared L3 and seven physical cores
plus one SMT sibling at 8T, so this is a diagnostic and not formal campaign
acceptance.

MKL is timed through its public LAPACKE API only. No vendor internals or vendor
source are consulted; no numerical implementation is translated in this work.
The reference LAPACK sources in `extern/OpenBLAS/lapack-netlib` are BSD-3 and
are the algorithmic record already cited in the SVD modules; they were read for
the shift/deflation structure, and no constant was carried over without a
measurement on this code.

## Investigation

Temporary stage timers in the blocked panel (removed before the commit) split
the 511 f64 values call into its stages at 1T and 8T. Before the change:

| stage | 1T ms | 8T ms |
| --- | ---: | ---: |
| `Y` product (`A^H u`) plus corrections | 5.15 | 2.27 |
| `X` product (`A r^H`) plus corrections | 5.40 | 2.80 |
| row gather/update/reflect | 0.70 | 1.16 |
| trailing update (two `k = 16` products) | 2.60 | 0.79 |
| column update | 0.26 | 0.35 |
| unblocked tail | 0.26 | 0.39 |
| compact-factor upgrade | 0.23 | 0.21 |
| dqds | 2.48 | 2.50 |
| copy | 0.06 | 0.06 |

The two products are memory-bound (80 GB/s at 1T, 237 GB/s at 8T) and cannot be
fused: `X_i` needs `r_i`, which needs the row update, which needs `Y_i`. The
remaining lever was the layout of the panel's rows and the trailing update.

Rejected, with the number that rejected it (local probe, median of 9):

- Hand-fused correction recurrences: 1T 18.34 against 17.17 ms, 8T 12.60
  against 10.77. The generic `matmul` kernels vectorize better than the
  hand-written `z!` loops.
- Initial dqds zero-shift sweep removed (`have_sweep = true`): 1T 17.16 against
  17.05, 8T 10.84 against 10.77. The shift from the raw factors costs more
  sweeps than it saves.
- 2/3 `dmin` shift variants: 508K and 911K entry updates against 507K.
- Values-only divide-and-conquer instead of dqds: 1T 22.99, 4T 13.11, 8T 13.85
  against 17.16 / 10.23 / 10.76. dqds stays.
- Iterator-formed dqds sweep and `fmin` in the sweep: 1T 15.96 and 15.78
  against 15.68. Only the pre-sweep `dmin` scan became branchless.

Adopted, in `faer/src/linalg/svd/bidiag.rs`, `dqds.rs` and `mod.rs`:

1. **The panel's rows are stored transposed.** A row of `A` has stride `m`,
   and the right reflectors stored in it are read again by the next steps'
   correction recurrences and by the trailing update. The panel now gathers
   each row once, reflects it into a contiguous `rows` buffer (one row per
   column, so a row is contiguous) and reads it from there. `A`'s rows are
   written back only when the compact factors are wanted, so the
   singular-value-only path skips the strided scatter.
2. **The values-only SVD skips the compact-factor upgrade** (`H_left`/
   `H_right` are never read). `BidiagParams::compute_householder_factors`
   defaults to true; `svd_imp` clears it when both output vectors are `None`.
3. **One trailing product of width `2 ib`.** `W = [V | X]` and `Z = [R^H | Y]`
   are assembled in the second half of the correction storage and applied as
   one product, so the trailing block is read and written once per panel. The
   copy is `3 (m + n) ib` scalars per panel against one traversal of the block.
4. **dqds keeps only whether the smallest pivot is the bottom one**, not its
   index, so the pre-sweep scan has no unpredictable branch per entry.

## Verification

- `cargo test -p colmat --release --lib`: 823 passed, 0 failed, 162 ignored.
  `linalg::svd`: 79 passed. The all-ones rank-1 case (`test_special`) stays
  under its tolerance; the transposed row buffer stores the same values as the
  scattered rows, so the vector paths are numerically unchanged.
- The Docker diagnostic verifies the whole timed grid at 1/4/8T before timing
  and checks the measured source identity again at the end.

## Paired result (candidate / baseline, below 1 is faster)

`scripts/docker_mkl.sh exec` with `docker/compare_svd.py --shapes
511,512,513 --families svd_values,svd_thin`, 3 process starts, 5 samples,
250 ms priming, MKL arm. 72 paired cells; the MKL control over all cells is
median 1.003, min 0.977, max 1.087 (the max is `svd_thin f64 F 511 8T`; that
cell's faer ratio 0.935 agrees with its neighbours, so the control moved, not
the candidate).

`svd_values`, median over C/F layout:

| dtype | shape | 1T | 4T | 8T |
| --- | --- | ---: | ---: | ---: |
| f64 | 511x511 | **0.901** | **0.804** | **0.813** |
| f64 | 512x512 | 0.909 | 0.809 | 0.822 |
| f64 | 513x513 | **0.901** | **0.803** | **0.815** |
| c64 | 511x511 | 0.941 | 0.867 | 0.858 |
| c64 | 513x513 | 0.933 | 0.858 | 0.850 |

`svd_thin` (guard), median over C/F layout:

| dtype | shape | 1T | 4T | 8T |
| --- | --- | ---: | ---: | ---: |
| f64 | 511x511 | 0.967 | 0.929 | 0.934 |
| f64 | 513x513 | 0.968 | 0.934 | 0.938 |
| c64 | 511x511 | 0.983 | 0.955 | 0.955 |
| c64 | 513x513 | 0.986 | 0.958 | 0.958 |

f64, milliseconds, 511/513:

| shape | layout | 1T | 4T | 8T |
| --- | --- | --- | --- | --- |
| 511 | F | 17.017 -> 15.332 | 10.120 -> 8.122 | 10.522 -> 8.601 |
| 511 | C | 17.074 -> 15.388 | 10.112 -> 8.151 | 10.611 -> 8.585 |
| 513 | F | 17.299 -> 15.591 | 10.261 -> 8.252 | 10.713 -> 8.736 |
| 513 | C | 17.457 -> 15.724 | 10.327 -> 8.274 | 10.780 -> 8.772 |

The f64 values-only case is 9.9% faster at one thread, 19.6-19.7% at four and
18.3-18.5% at eight. Complex f64 is 6-7% at one thread and 14-15% at four and
eight: its bidiagonalisation is twice the element work, so the serial stages
this change removes are a smaller share of its total.

## What remains unverified

- f64 values-only at one thread is 9.9% (ratio 0.9007-0.9013 across the four
  cells), just under the 10% mark, and complex is 6-7%; the four- and
  eight-thread results are past it for both. The remaining one-thread f64 time
  is 13.3 ms of bidiagonalisation (8.9 of it the two memory-bound products) and
  2.5 ms of dqds. The candidates for the last percent (an FMA on the scalar
  dqds chain, a values-only two-way divide-and-conquer merge) are not
  implemented.
- QR at 511/513 was not changed. The local probe's width scan put the QR
  factorization at 16 columns about 9% ahead of the shipped 32 at 8T and flat
  at 1T; that is a lead with no paired run.
- The host cannot establish formal adoption (boost on, shared L3, no 64-core
  scaling). A campaign pair remains for whoever moves the pin.

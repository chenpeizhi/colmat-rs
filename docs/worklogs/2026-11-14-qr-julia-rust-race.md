# QR Julia/Rust competition

## Decisions

The target is 2048×256 f64, eight threads, complete thin Q/R. Julia must beat
current colmat by at least 1.05× before transferring an improvement to Rust.
Repeat until Rust's whole-call time is at least 10% below the original baseline.
The baseline production code is b292f474; no upstream revision was measured.
The experimental branch starts at 0bbf72c9.

All variants use identical seeded input, with allocations inside the owning-call
timing, on CPU set `1,3-9`. CPU 2 showed unrelated contention; CPU 17 is occupied
by the user. Runtime creation, validation and preparatory GC are outside timing.
Separate Rust and Julia processes avoid the mixed-runtime order effects seen in
[the earlier comparison](2026-11-14-colmat-julia-comparison.md).

Julia uses recursive Householder panels and colmat's inverse compact-WY
coefficients. The useful design is small factor panels (16) and larger Q panels
(64), rebuilt from the packed reflectors and tau inside timing. Q construction
skips known zero columns and directly initializes each panel's identity columns.
The selected Julia arm uses BLAS eight-thread products and serial panel control;
the ThreadTeams variants use single-thread BLAS. These are alternative parallel
backends, never nested active eight-thread teams. General EVD/SVD and old qr_lab
are unchanged. Julia is a Float64 algorithm experiment, not a bitwise translation.

The transferable zero-prefix proof: during reverse application to an identity
block with seed row offset s, columns j with s+j preceding the current panel
have zero support in the active rows. They require no update. This property
belongs to identity-seeded construction; arbitrary-input multiplication cannot
use it. Restricting the target adds no buffers or parallel regions. Existing
column partitioning still reads the panel per participating task: its shared
operand traffic remains O(p*m*b), pending many-core measurement.

## Rejected candidates and evidence

Frozen sources and complete results are in the sibling colmat-benchmark tree,
under `data/probes/` (the first prototype is under `probes/qr-julia-rust-race/`).
These are scans on witch, not formal campaign acceptance.

- `first-alignment`: aligned Julia 37.45 ms, prefix 35.41 ms, Rust 24.04 ms.
  Native phase profile: factor 14.59 ms, Q 8.23 ms; Q merits specialization.
- `20261114-qr-race-alignment-scan`: coefficient reuse and column-grain scan;
  best prefix 28.75 ms versus Rust 20.74 ms, rejected.
- `20261114-qr-race-merge-scan`: hierarchical coefficient merging/padding;
  best prefix 29.60 ms versus Rust 20.78 ms, rejected.
- `20261114-qr-race-leaf-scan`: larger serial leaves; best total 24.45 ms
  versus Rust 20.92 ms, rejected.
- `20261114-qr-race-noscale-pilot`: omitting redundant whole-input scaling
  appeared to win (22.10 versus 23.47 ms). Mixed-runtime three-start confirmation
  `20261114-qr-race-julia-confirm` failed: Julia 26.00/25.67/22.52 ms versus
  Rust 24.05/24.15/20.63 ms. No production Rust change followed that pilot.
- `20261114-qr-race-structured-pilot` and `...-structured-blockscan`: identity
  structure alone insufficient. BLAS-parallel block 32 pilot 20.69 versus
  Rust 23.09 ms; isolated `...-structured-confirm` failed the 1.05× criterion
  in two starts (20.85 versus 20.94; 20.55 versus 20.96 ms).
- `20261114-qr-race-reblock-scan`: independently scan factor/Q widths;
  16/64 selected at 19.08 versus Rust 22.86 ms. Rebuilding coefficients is included.

## Verification conclusions and constraints

`20261114-qr-race-reblock-confirm` fixes 16/64 before three fresh isolated-process
starts, eleven repetitions each, alternating backend order. Julia/Rust medians
in ms: 19.449/20.891, 16.619/23.061, 19.784/22.787. Speedups 1.074×, 1.388×,
1.152×; all exceed 1.05×. Rust A/A labels agree within 0.4%. All six process
affinity audits and numerical output checks pass. Raw load observations and
outliers are retained. This establishes a local Julia lead, not a language
ranking, saturation limit, or portable adoption result.

The Julia QR tests pass 566 assertions: recursive/tail panels, zero/rank loss,
empty/tall/wide, extreme scale, poisoned unused coefficient triangles, panel
reblocking, serial and ThreadTeams execution. Every timed output is additionally
checked for reconstruction and orthogonality. The no-prescale path follows the
Rust owning QR boundary; near-overflow behavior is not established by these tests.

The first Rust transfer skips zero-prefix columns in identity-seeded QR/LQ Q
construction and the owning Qr methods. Generic application is unchanged.
`20261114-qr-race-rust-prefix-confirm` retains three flanked baseline/candidate
starts and guard cells: target whole-call ratios 1.010/1.072/0.960, insufficient
for the target; Q alone fell from about 8.3 to 6.9 ms. Native factor timing also
varied between processes despite unchanged factor code. The 1T guard fell from
34.05 to 29.48 ms; 1024×128/8T and 256-square/8T stayed within 1%. Julia still
leads the candidate at 19.62/19.18/19.59 versus 22.04/22.08/22.03 ms.
This transfer is a prerequisite experiment, not an 8T improvement claim.
825 library tests pass (162 ignored), including four-scalar reconstruction,
orthogonality, old-sequence oracle, row-major output and tail panels. The
process-wide allocation test passes, including the new 4T/8T thin-Q calls.

The next hypothesis is to avoid projecting a panel against columns that are
still its identity seeds, as the Julia candidate does. The 10% target remains
pending. Rust changes require four-scalar
focused checks, the full library suite and low-level allocation checks before
timing. This single-L3 shared host cannot satisfy the formal campaign's separate
observer-cache requirement. 64-core scaling and formal adoption remain unverified.


The direct identity-seed Rust candidate was dropped. Its source is frozen in
`20261114-qr-race-rust-seeds-paired` and `...-rust-seeds-confirm`. A same-binary
control reconstructing the original owning Q path gave target ratios
0.893/0.909/0.897, but comparison with the frozen original binary gave only
0.948/0.957/0.950. The 256-square/8T guard worsened from 2.93 to 3.43 ms
(Q stage 0.710 to 0.768 ms); 1024×128 improved 6.18 to 5.73 ms and the 1T
target 35.58 to 28.00 ms. The target and small-square guard do not justify
carrying this extra specialization. Its full 825-test suite, allocation and
no_std checks passed; rejection is performance-based. The same-binary control
remains a diagnostic, not a replacement for frozen-binary acceptance.

Next probe: expose the recursive panel's leaf width as an explicit parameter
(default preserving the original width), scan it using low-level QR on the
same input/team, and only then consider the owning tall-QR choice. This
transfers Julia's measured small serial panels without changing generic Q
application or hard-capping a parallel region.

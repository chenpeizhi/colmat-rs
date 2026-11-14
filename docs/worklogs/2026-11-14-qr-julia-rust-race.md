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

Rust transfer and 10% target remain pending. Rust changes require four-scalar
focused checks, the full library suite and low-level allocation checks before
timing. This single-L3 shared host cannot satisfy the formal campaign's separate
observer-cache requirement. 64-core scaling and formal adoption remain unverified.

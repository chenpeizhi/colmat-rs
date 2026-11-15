# QR Julia/Rust competition

## Decisions

The local target is 2048×256 f64, eight threads, the complete allocation-inclusive
call producing thin Q and R. Julia must first beat current Rust by at least
1.05×; transfer improvements to Rust until its whole-call time is at least 10%
below the original Rust. The production baseline is b292f474 (the branch starts
at 0bbf72c9, with identical production source). No upstream revision was measured.

The final Rust experiment carries two changes:

- Identity-seeded Q construction skips columns whose seeds precede the active
  reverse Householder panel. It serves owning Qr, low-level QR unitary and the
  shared LQ identity-seeding path. Arbitrary-input Householder application is
  unchanged. The skipped columns have exactly zero support in the active rows;
  this property does not apply to general matrix inputs.
- Owning Qr uses factor/Q blocks of 16 and recursive leaves of 8 when n≤256,
  m≥2n, and the original recommended block exceeds 16. Other shapes retain the
  original factor policy. The leaf is a crate-private argument of the factor
  entry (`qr_in_place_with_leaf`), not a `QrParams` field: no measured case asks
  a caller to choose it, so `QrParams` is unchanged and the public low-level QR,
  LQ and SVD calls keep the original recursion (leaf 1). The caller still owns
  the team; products use their cost-selected budget without a fixed thread cap.

Julia's selected configuration uses factor panels 16, Q panels 64, identity
structure, and eight-thread BLAS under serial panel control. Alternative
ThreadTeams variants use BLAS=1. The Rust winner uses 16 for both stages:
regrouping coefficients into larger Q panels cost more than it saved. Thus the
competition transfers useful structure and tests parameters in each backend.
The old qr_lab, eigh_lab and svd_lab entrypoints remain available.

## Verification conclusions and constraints

The local 10% target is achieved. Final frozen-binary measurements are in sibling
`colmat-benchmark/data/probes/20261114-qr-race-rust-block16-confirm/`, with frozen
sources, both libraries, hashes, controller, all samples, process summaries,
paired summaries and verification logs. This is machine-local probe evidence,
not a published acceptance campaign.

The leaf is asserted where reconstruction cannot: `recursion_leaf_size_controls_the_blocking_depth`
runs the recursion sequentially and counts its nodes, so a leaf value that never
reaches the decision (equal counts across leaves) fails the test. The
reconstruction and orthogonality sweep over leaf widths remains, but it is not
what establishes engagement. The `no_step_alloc` gate no longer sets the removed
field: it exercises the identity-seeded Q path with the default recursion, which
is what its allocation claim covers.

Each start launches baseline-before, candidate, baseline-after and Julia in
separate processes, eleven repetitions each. Native A/A labels call identical
code. The pairing is candidate median divided by the geometric mean of its
flanking baseline medians, fixed in the controller declaration.

| Start | Original Rust, flanks' geometric mean (ms) | Improved Rust (ms) | Julia (ms) | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| 0 | 23.018 | 14.586 | 19.451 | 36.63% |
| 1 | 21.777 | 18.070 | 19.165 | 17.02% |
| 2 | 23.032 | 14.646 | 19.510 | 36.41% |

Median reduction is 36.41% (1.573×). All three exceed 10%, even against the
faster individual flank (worst ratio 0.8794). Target A/A medians differ by at
most 0.53%. Start 1's baseline drifts 23.079→20.548 ms; retain this variation
and the candidate's 14.6–18.1 ms range rather than treating 36% as a portable
expectation. The improved Rust also beats Julia in each final cohort.

Reference guards, one process per binary and seven repetitions: 2048×256/1T
33.909→27.271 ms; 1024×128/8T 6.127→5.606 ms; 256-square/8T 3.349→2.970 ms.
The small-square candidate's A/A labels differ by 10.54%, so its apparent
improvement is unresolved, not a positive performance conclusion. No guard
shows a measured regression, but these scans do not establish formal acceptance.

All 315 timed outputs pass reconstruction/orthogonality validation, and all 18
process affinity audits pass. The final library suite passes 827 tests, with
162 ignored. Focused tests cover f32/f64/c32/c64, recursion leaves 1..48,
two panels plus tails, zero/rank loss, row-major borrowed input/output, thin/full
Q and complement, policy boundaries at 256/257 columns and aspect 2, and repeat
Q construction at a fixed count. Serial leaf sweeps passed before the held-team
sweeps; final owning-policy tests use a held team of four. The process-wide
allocator test passes: factor block16/leaf8 and thin Q allocate nothing after
warm-up on held teams of four/eight with preallocated buffers. Owning calls
intentionally allocate; their allocation counts are not reported as zero.
The no-default-features library check passes. Julia passes 566 new QR assertions
and 1,170 existing laboratory assertions.

All measurements use identical input bytes (seed 84, fixture draw order retained)
and CPU IDs `1,3-9`, eight physical cores within the user's 1..16 limit. CPU 2
showed contention; CPU 17/35 are excluded. Pool creation, validation copies,
preparatory GC and a 20 ms settle are outside timing; input/output/workspace
allocation is inside. Runtime and BLAS arms run in separate processes after
[the earlier comparison](2026-11-14-colmat-julia-comparison.md) exposed order
sensitivity. Phase profiles are separate calls and must not be subtracted from
whole-call medians as if they shared allocator/cache state.

This is a shared, single-L3 host with turbo enabled. All 18 broad endpoint idle
flags fail (they conservatively include CPU 2 and the full 1..16 domain and SMT
siblings); no failed observation was retried away. The target comparison has
stable within-process A/A controls but substantial between-process variation.
The experiment remains EXPERIMENT: 64-core/NUMA scaling, every affected campaign
shape, four-dtype performance, independent observer cache and formal vendor
acceptance remain unverified. Shapes beyond the measured 512..4096 rows and
128/256 columns are not established by the selected policy.

The leaf adds serial O(m*n*leaf) panel work. Existing column splits still read
the panel per participating task, O(p*m*b) shared-operand traffic; no new
per-thread full matrix or serial combine is introduced. Zero-prefix restriction
adds no scratch, and changing panel/leaf sizes uses existing scratch queries.
No proprietary implementation was inspected or used as algorithmic source.

## Rejected candidates and selection evidence

Complete sources/results live in sibling `colmat-benchmark/data/probes/` unless
otherwise noted. These are scans; only the final three-cohort frozen-binary
comparison above establishes the local target.

- Initial inverse-WY alignment: 37.45 ms versus Rust 24.04; skipping zero Q
  columns gave 35.41. Frozen under `probes/qr-julia-rust-race/first-alignment`.
  Native phase diagnostics (factor 14.59, Q 8.23 ms) justified investigating Q.
- `20261114-qr-race-alignment-scan`, `...-merge-scan`, `...-leaf-scan`:
  coefficient reuse, hierarchical merging, grains/padding and larger serial
  leaves did not win; best totals 28.75/29.60/24.45 ms versus respective Rust
  20.74/20.78/20.92 ms. Retain these failures rather than retrying them later.
- `...-noscale-pilot` appeared to win after omitting redundant whole-input
  normalization. Mixed-runtime `...-julia-confirm` failed: Julia
  26.00/25.67/22.52 versus Rust 24.05/24.15/20.63 ms. No Rust change followed
  that pilot. `...-structured-pilot`, `...-structured-blockscan` and isolated
  `...-structured-confirm` also failed the 1.05× criterion in two starts.
- Julia `...-reblock-scan` selected 16/64. `...-reblock-confirm` fixed it before
  three isolated starts: Julia 19.449/16.619/19.784 versus Rust
  20.891/23.061/22.787 ms; speedups 1.074/1.388/1.152×. This established the
  required Julia lead before the first Rust transfer. Its no-prescale path's
  near-overflow behavior remains unverified; extreme-scale fixtures pass.
- Rust `...-rust-prefix-confirm`: Q fell approximately 8.3→6.9 ms, but target
  whole-call ratios 1.010/1.072/0.960 did not establish an 8T improvement alone.
  The 1T guard improved 34.05→29.48 ms; small 8T guards stayed within 1%.
- Direct identity-seed Rust specialization was dropped. Its source is frozen
  in `...-rust-seeds-paired` and `...-rust-seeds-confirm`. Same-binary original-Q
  controls suggested ratios 0.893/0.909/0.897, but frozen-binary ratios were
  0.948/0.957/0.950 and the square guard worsened 2.93→3.43 ms. Numerical,
  allocation and no_std checks passed; rejection was performance-based. The
  diagnostic original-Q control remains; after changing factor policy it is
  an original-Q control only, not an original whole-QR baseline.
- `...-rust-leaf-scan`: root48 factor medians for leaves 1/4/8/12/16/24/48
  were 12.342/12.344/11.963/11.589/11.596/12.242/16.369 ms. Owning leaf12
  pilot `...-rust-leaf12-pilot` reached ratio 0.930, short of the target.
  The unconditional owning leaf12 policy was replaced by the final scoped one.
- `...-rust-reblock-scan`: factor/Q/leaf 16/16/8 wins in all three starts
  (whole RawQr 16.82/16.79/14.72 ms), against 48/48/12
  (20.10/20.17/18.08). Regrouping to 16/64/8 cost 17.96/17.87/15.80 ms;
  32/32/8 and unblocked leaf16 also lost. RawQr differs from owning Qr and is
  used only to select parameters. Its controller inherited factor-only wording;
  a selection clarification retaining whole-call cost was recorded before
  inspecting results, and is preserved alongside it.
- `...-rust-shape-scan` checks the selected tuple at 1024×128, 512×256,
  1024×256 and 4096×256 (raw total reductions about 12/16/3/15% against the
  original factor policy under the same Q constructor). Square and 512-column
  probes do not broaden the final policy; their full results remain recorded.

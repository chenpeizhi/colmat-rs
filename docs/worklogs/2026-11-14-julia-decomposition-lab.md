# Julia decomposition laboratory

## Decisions

- User requested minimal Julia implementations under `./julia` to investigate
  single- and multi-thread headroom in colmat eigh, QR and SVD. The code is a
  separate local branch, `experiment/julia-headroom`, based on b292f474. The
  solver-only Rust branch remains unchanged and was pushed separately.
- Float64 only: independent conventional Householder implementations, blocked
  compact-WY QR and basis construction, symmetric tridiagonal reduction,
  and deferred two-sided blocked bidiagonal reduction (width16, floor128,
  padded leading dimension). The weight/column repair arithmetic follows
  Rust b292f474. No proprietary implementation internals were consulted.
- Terminal tridiagonal eigenproblems and bidiagonal SVDs use Julia's standard
  solver. This deliberately does not reproduce Rust's secular root finder,
  D&C recursion, dqds, SIMD, fused lower-triangle update, QR look-ahead or
  cost model. The README maps every boundary; comparisons do not establish
  the performance ceiling of these omitted Rust components.
- ThreadTeams 1.4.0 owns persistent workers; disjoint output chunks use
  foreach_range. Custom kernels use one-thread BLAS. The standard dense
  comparisons set BLAS to the tested width. There is no nested team-budget
  splitting equivalent to Rust. Worker-local full matrices and serial partial
  result combination are avoided; repeated reads/packing of a shared small
  basis can nevertheless grow with p. 64-core scaling is unverified.
- Initial per-reflector basis construction was replaced by compact-WY blocks
  to match Rust's block Householder apply design. In the first noisy scan,
  2048x256 QR spent150.7ms building Q versus20.0ms factoring at1T. This was
  an artificial reference-implementation bottleneck, not a Rust limitation.
  Both source snapshots and all measurements are retained.
- High-level drivers allocate outputs and workspaces; low-level destructive
  kernels reuse caller-owned workspaces. These are distinct measurement
  contracts. The scripts live here because the user explicitly requested the
  Julia experiment under `./julia`; the normal Rust campaign stays in the
  sibling colmat-benchmark repository.
- The `./julia` tree was removed from the fork on 2026-11-15 (`PATCHSET.md`
  records the drop). It was reference scaffolding with no workspace dependency
  and the fork ships no Julia path; the source stays in history (`bfee09e5`)
  and on `experiment/julia-headroom`.

## Verification conclusions and constraints

- All1,170 assertions passed. Residual/reconstruction, orthogonality, ordering, empty/rectangular,
  rank-deficient/zero, clustered/graded spectra, extreme global scale,
  block/floor/padding boundaries, repair bit identity and kernel inference
  were checked with Serial and Team(16), confined to CPU IDs1..16.
- The final numerical validation also checks exactly the timed n256/512
  input seed, shape and draw order, including 1024x128/2048x256 QR. This
  broader timed-input check was added after the descriptive scan; it is not
  being presented as a predeclared acceptance arm.
- Each scan has616 timed calls: seven samples for11 cases at two sizes and
  four budgets (1/4/8/16). Warm-up, matrix reset and explicit GC are excluded
  from sample times. Full-call timings include allocation and normalization;
  per-stage times exclude setup. All280 final kernel/repair samples counted
  zero allocated bytes. Whole drivers allocate; no global allocation-free
  API contract is claimed.
- The Linux launcher holds the sibling campaign lock, observes load before
  and after, retains failed guards, and audits Julia thread affinity.
  Final observed thread masks were exclusively1..16; no Julia worker used
  CPU17/35. Observer runs on CPU0. CPU1..16 are16 separate physical cores;
  their SMT siblings19..34 and the shared L3 still affect timings.
- Both scans failed the endpoint load conditions: CPU2 was roughly50% busy
  before/after the final scan. No reruns were performed to obtain a clean
  guard. Ratios are descriptive and must not be used for acceptance or a
  claim about Rust headroom. Budgets were traversed in one process, not a
  randomized paired campaign. A standard QR16T outlier and inconsistent
  tridiagonal16T timings further limit interpretation.
- No upstream revision or same-input Rust baseline was measured in this
  experiment. The existing Rust b292f474 repair result is a separate record.
  The useful outcome here is a runnable, validated stage-by-stage laboratory,
  with serial/parallel execution and explicit algorithm boundaries.

Detailed local evidence in the sibling colmat-benchmark checkout:

- `data/probes/20261114T084303Z-julia-headroom-scalar-basis/`
- `data/probes/20261114T084303Z-julia-headroom-blocked-basis/`

Each includes all samples, summary, environment and load observations, frozen
Julia source and SHA256 hashes. Final validation evidence is stored separately
under the latter record's validation directory. No benchmark page was published.

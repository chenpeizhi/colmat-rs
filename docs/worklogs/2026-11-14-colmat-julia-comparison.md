# Same-input colmat versus Julia comparison

## Decisions

Compare Float64 colmat (Rust source b292f474), the Julia laboratory and Julia
standard dense solvers. A private cdylib calls the Rust public owning APIs;
no production path or supported C API changes. Each backend regenerates exactly
the same seeded matrices, checked by SHA256. Outputs are complete eigenpairs,
thin QR including Q, and thin SVD including explicit V. Factor/workspace
allocation is timed; pool startup, validation copies, final-output destruction,
preparatory GC and20ms settle are not. Rust heap bytes are marked missing,
not misreported as zero by Julia allocation counters.

Use a separate process per backend/budget. The initial mixed-process scan was
rejected because of implementation-order dependence: 2048x256 QR16T Rust calls
were around15ms after custom Julia but55–58ms after standard Julia. Lingering
BLAS workers are a plausible mechanism: the inspected OpenBLAS reference thread
server can spin longer than20ms. The exact JLL timeout was not measured, so this
is not a proven attribution. Process separation removes cross-backend workers
without tuning the vendor runtime. OPENBLAS_NUM_THREADS=1 at startup; custom
Julia keeps BLAS=1, standard Julia sets BLAS=width. No unused Rust/ThreadTeams
pool is started in the other backends.

Grid: eigh/SVD256/512, QR1024x128/2048x256; 1/4/8/16T; three starts per backend;
five samples per cell. CPU IDs1..width, excluding17/35. Backend process order
rotates, budget order reverses between starts. Aggregate the per-process
medians, then Julia/Rust ratios within matched starts, then median over starts.
Keep all samples, failed load checks and min/max process medians.

## Verification conclusions and constraints

All1,080 timed outputs passed residual/orthogonality and spectrum ordering checks;
maximum residual4.12e-15, maximum orthogonality error2.49e-12. All36 processes
passed affinity checks. Small rectangular/wide/zero FFI cases and exact timing
inputs were checked before their measurements. The bridge lock introduces no
new dependency version except its own package. No upstream revision was timed.

Only1/36 processes passed both endpoint load observations. In particular, CPU2
became heavily loaded during the resumed scan, with observed use up to95%.
The8T colmat512 EVD medians ranged27–85ms; standard Julia ranged46–200ms.
These are characterization data, not adoption evidence or a universal ranking.
The user-interrupted first attempt of s0-t16-julia_stdlib produced no samples;
its empty partial output is archived under interrupted/. Eleven completed
processes were retained, source/library hashes checked unchanged, and only
unfinished cases resumed. This also leaves a temporal gap in the first16T cohort.

Single-thread results were less variable: median512 EVD colmat71.89ms,
custom Julia84.92ms, standard Julia56.47ms; QR2048x25635.79/50.44/39.21ms;
SVD512125.95/375.87/116.44ms. Complete results and process spreads belong in
the evidence rather than being treated as accepted speedups.

This compares implementations, including algorithms and BLAS. Julia1.13.3's
standard symmetric eigen solver defaults to RobustRepresentations/syevr (MRRR);
colmat uses D&C. The lab's explicit boundaries remain in julia/README.md.
The serial EVD gap is a lead for investigation, not proof of a compiler limit.

Local evidence in the sibling colmat-benchmark checkout:

- data/probes/20261114-colmat-vs-julia/: rejected mixed-process scan, frozen source.
- data/probes/20261114-colmat-vs-julia-isolated/: all samples, per-process and final
  summaries, numerical errors, affinity/load observations, frozen sources and
  compiled bridge with SHA256, plus interruption/resume records.

# `norm_l2` at the floating-point range limits

## Decisions

- Keep the existing three branchless SIMD accumulators and pairwise reduction.
  Widen the reciprocal scales from the square roots of the range limits by a
  factor of `eps`, as the crate's complex absolute value already does. Choose
  the result by comparing the *unscaled* sum with the old square-root bounds.
- For the native types, writing $L$ for `MIN_POSITIVE` and $\epsilon$ for machine
  epsilon, the smallest subnormal is $L\epsilon$. Scaling it up by
  $1/(\sqrt{L}\epsilon)$ before squaring gives $L$, a normal value. Near the
  upper limit, scaling down by $\sqrt{L}\epsilon$ before squaring leaves room
  for any representable norm. At the two selection cutoffs, the respective
  rescaled sums are still well inside the normal range.
- A strided view now accumulates the same three sums without SIMD. Feeding
  subnormal values to a running `hypot` rounds each intermediate norm:
  repeated copies of the smallest subnormal can get stuck at one copy. Scale
  before squaring the entries, not after computing their norm.

## Verification conclusions and constraints

- The regression test covers real and imaginary inputs in both native
  precisions, contiguous and strided views, the pairwise split, subnormal
  inputs, finite norms whose squares overflow, true overflow, and NaN mixed
  with infinity. The issue's singleton near overflow returns infinity before
  the fix.
- No performance comparison or upstream run. The separate QDWH workaround in
  [#226](https://github.com/tensor4all/colmat-rs/pull/226) can be revisited once
  this change lands; its scaling also protects downstream QR calculations.

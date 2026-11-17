//! the $RQ$ decomposition decomposes a matrix $A$ into the product
//! $$A = RQ$$
//! where $R$ is an upper trapezoidal matrix whose last column ends on the last
//! row (for $m \le n$ the right $m \times m$ block of $R$ is upper triangular
//! and the columns before it are zero) and $Q$ is a matrix with orthonormal rows
//! (represented using block householder sequences).
//!
//! the factorization is the $LQ$ decomposition of $A$ with its rows and its
//! columns reversed, $P_m A P_n = \tilde L \tilde Q$, read back through the same
//! reversal: $A = (P_m \tilde L P_m)(P_m \tilde Q P_n)$. `factor::rq_in_place`
//! runs [`lq_in_place`](crate::linalg::lq::no_pivoting::factor::lq_in_place) on
//! `A.reverse_rows_and_cols_mut()`, so the triangular factor lands in the upper
//! right of $A$ and the householder vectors of $Q$ along the rows to its left,
//! the storage layout of LAPACK's `?gerqf`. `unitary` builds $Q$ (thin or
//! full), `reconstruct` rebuilds $A = RQ$, and `factor::copy_r` copies $R$ out.
pub mod no_pivoting;

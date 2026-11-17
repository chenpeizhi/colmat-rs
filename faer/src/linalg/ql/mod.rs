//! the $QL$ decomposition decomposes a matrix $A$ into the product
//! $$A = QL$$
//! where $Q$ is a unitary matrix (represented using block householder
//! sequences) and $L$ is a lower trapezoidal matrix whose last row ends on the
//! last column: for $m \ge n$ the bottom $n \times n$ block of $L$ is lower
//! triangular and the rows above it are zero.
//!
//! the factorization is the $QR$ decomposition of $A$ with its rows and its
//! columns reversed, $P_m A P_n = \tilde Q \tilde R$, read back through the same
//! reversal: $A = (P_m \tilde Q P_m)(P_m \tilde R P_n)$. `factor::ql_in_place`
//! runs the blocked $QR$ kernel on `A.reverse_rows_and_cols_mut()`, so the
//! triangular factor lands in the lower right of $A$ and the householder vectors
//! above it, the storage layout of LAPACK's `?geqlf`. `unitary` builds $Q$ (thin
//! or full), `reconstruct` rebuilds $A = QL$, and `factor::copy_l` copies $L$
//! out.
pub mod no_pivoting;

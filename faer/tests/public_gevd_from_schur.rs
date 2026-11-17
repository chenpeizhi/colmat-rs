//! tensor4all/colmat-rs#211: eigenvectors of an existing generalized Schur
//! pencil.
//!
//! `gevd_real` / `gevd_cplx` own the whole reduction, so the vectors of a saved
//! `(S, T)` could not be obtained without redoing the QZ: the issue's MWE failed
//! with `E0603`, both conversion routines were private. `gevd_from_schur_real` /
//! `gevd_from_schur_cplx` take the saved pencil read-only, keep its diagonal
//! order including infinite eigenvalues, and optionally back-transform by `Q` /
//! `Z`.
//!
//! The residuals here are `||beta*A*x - alpha*B*x||` and the left counterpart,
//! relative to `||x|| (|beta| ||A|| + |alpha| ||B||)`, computed in `c64` after
//! normalizing the vector and the homogeneous pair first (see
//! `tests/gevd_infinite_eigenvalue.rs` for why the normalization cannot be
//! skipped). A real pencil's left vectors satisfy the transpose relation
//! `y^T A = lambda y^T T` (what `qz_to_gevd_real` produces, pinned by
//! `linalg::gevd::tests::left_vectors_of_schur_pencil_f64`), a complex pencil's
//! the adjoint one.
mod common;
use common::*;

use colmat::dyn_stack::StackReq;
use colmat::linalg::gevd::{
	self, ComputeEigenvectors, gevd_from_schur_cplx, gevd_from_schur_real,
	gevd_from_schur_scratch, schur_eigenvalues_real,
};
use colmat::{MatRef, Scale};

/// the imaginary unit
fn i() -> c64 {
	c64::new(0.0, 1.0)
}

fn close64(x: c64, y: c64, tol: f64) -> bool {
	(x - y).norm() <= tol * y.norm().max(1.0)
}

fn as_c64<T: ComplexField>(
	m: MatRef<'_, T>,
	up: &impl Fn(&T) -> c64,
) -> Mat<c64> {
	Mat::from_fn(m.nrows(), m.ncols(), |r, c| up(&m[(r, c)]))
}

/// the encoded vector of column `j`: one column, or a conjugate pair as one
/// complex vector `col_j + i col_{j+1}`
fn eigenvector<T: ComplexField>(
	m: MatRef<'_, T>,
	j: usize,
	nw: usize,
	up: &impl Fn(&T) -> c64,
) -> Mat<c64> {
	Mat::from_fn(m.nrows(), 1, |r, _| {
		if nw == 1 {
			up(&m[(r, j)])
		} else {
			up(&m[(r, j)]) + i() * up(&m[(r, j + 1)])
		}
	})
}

/// relative left and right residuals of one homogeneous eigenpair, normalized
/// first so that neither the numerator nor `|alpha| + |beta|` overflows.
/// `transposed` selects the left relation of a real pencil.
fn pair_residual(
	a: &Mat<c64>,
	b: &Mat<c64>,
	alpha: c64,
	beta: c64,
	u: &Mat<c64>,
	v: &Mat<c64>,
	transposed: bool,
) -> (f64, f64) {
	let (nu, nv) = (u.norm_l2(), v.norm_l2());
	assert!(
		nu > 0.0 && nu.is_finite() && nv > 0.0 && nv.is_finite(),
		"eigenvector norms {nu:e} / {nv:e}, alpha = {alpha:?}, beta = {beta:?}"
	);
	let scale = alpha.norm().max(beta.norm());
	assert!(
		scale > 0.0 && scale.is_finite(),
		"eigenvalue scale {scale:e}, alpha = {alpha:?}, beta = {beta:?}"
	);
	let (alpha, beta) = (alpha / scale, beta / scale);
	let (u, v) = (u / nu, v / nv);
	let (na, nb) = (a.norm_l2(), b.norm_l2());
	let den = beta.norm() * na + alpha.norm() * nb;
	assert!(
		den > 0.0 && den.is_finite(),
		"residual denominator {den:e}, ||A|| = {na:e}, ||B|| = {nb:e}"
	);
	let left = if transposed {
		u.transpose() * a * Scale(beta) - u.transpose() * b * Scale(alpha)
	} else {
		u.adjoint() * a * Scale(beta) - u.adjoint() * b * Scale(alpha)
	};
	let right = a * &v * Scale(beta) - b * &v * Scale(alpha);
	let (left, right) = (left.norm_l2() / den, right.norm_l2() / den);
	assert!(
		left.is_finite() && right.is_finite(),
		"residual {left:e} / {right:e}, alpha = {alpha:?}, beta = {beta:?}"
	);
	(left, right)
}

/// checks every column or pair of one saved pencil: residual, unit norm of the
/// encoded complex vector, and the unit column of a `0/0` block.
/// one diagonal block: first column, width, and the homogeneous pair
type Pair = (usize, usize, c64, c64);

#[allow(clippy::too_many_arguments)] // one flat test helper, not a kernel
fn check_pairs<T: ComplexField>(
	a: MatRef<'_, T>,
	b: MatRef<'_, T>,
	ul: MatRef<'_, T>,
	ur: MatRef<'_, T>,
	pairs: &[Pair],
	up: &impl Fn(&T) -> c64,
	tol: f64,
	transposed_left: bool,
) {
	let (ca, cb) = (as_c64(a, up), as_c64(b, up));
	for (k, &(j, nw, alpha, beta)) in pairs.iter().enumerate() {
		let (u, v) = (eigenvector(ul, j, nw, up), eigenvector(ur, j, nw, up));
		for (side, x) in [("left", &u), ("right", &v)] {
			let norm = x.norm_l2();
			assert!(
				(norm - 1.0).abs() <= tol,
				"pair {k} at column {j}: {side} vector norm {norm:e} is not one"
			);
		}
		if alpha.norm() + beta.norm() == 0.0 {
			// a nonregular 0/0 block: the unit vector `e_j`, without the
			// back-transformation (LAPACK `dtgevc`, see the rustdoc)
			for (side, x) in [("left", &u), ("right", &v)] {
				for r in 0..a.nrows() {
					let want = if r == j { 1.0 } else { 0.0 };
					assert!(
						close64(x[(r, 0)], c64::new(want, 0.0), tol),
						"pair {k} at column {j}: {side} column is not e_{j}"
					);
				}
			}
			continue;
		}
		let (left, right) =
			pair_residual(&ca, &cb, alpha, beta, &u, &v, transposed_left);
		assert!(
			left <= tol,
			"pair {k} at column {j} (alpha = {alpha:?}, beta = {beta:?}): \
			 left residual {left:e} > {tol:e}"
		);
		assert!(
			right <= tol,
			"pair {k} at column {j} (alpha = {alpha:?}, beta = {beta:?}): \
			 right residual {right:e} > {tol:e}"
		);
	}
}

/// 7x7 real generalized Schur pencil: `2/1`, a conjugate pair `1 +- i` over a
/// `1.5 * I` block, an infinite `4/0`, a repeated `2/1`, a second infinite
/// `-3/0`, and a nonregular `0/0` block. The entries above the blocks couple
/// them, so no vector is trivial.
fn saved_real_pencil() -> (Mat<f64>, Mat<f64>) {
	let n = 7;
	let mut s = Mat::<f64>::zeros(n, n);
	let mut t = Mat::<f64>::zeros(n, n);
	for j in 0..n {
		for r in 0..j {
			s[(r, j)] = 0.05 * ((r + 2 * j) % 5) as f64 - 0.1;
			t[(r, j)] = 0.03 * ((3 * r + j) % 7) as f64 - 0.09;
		}
	}
	s[(0, 0)] = 2.0;
	t[(0, 0)] = 1.0;
	s[(1, 1)] = 1.0;
	s[(1, 2)] = 2.0;
	s[(2, 1)] = -0.5;
	s[(2, 2)] = 1.0;
	t[(1, 1)] = 1.5;
	t[(1, 2)] = 0.0;
	t[(2, 2)] = 1.5;
	s[(3, 3)] = 4.0;
	s[(4, 4)] = 2.0;
	t[(4, 4)] = 1.0;
	s[(5, 5)] = -3.0;
	(s, t)
}

fn check_saved_real<TR: RealField>(up: impl Fn(&TR) -> c64, eps: f64) {
	let (sf, tf) = saved_real_pencil();
	let n = sf.nrows();
	let s = Mat::<TR>::from_fn(n, n, |r, c| from_f64(sf[(r, c)]));
	let t = Mat::<TR>::from_fn(n, n, |r, c| from_f64(tf[(r, c)]));
	let (mut ul, mut ur) = (Mat::<TR>::zeros(n, n), Mat::<TR>::zeros(n, n));
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<TR>(n, Par::Seq.shape()));
	gevd_from_schur_real(
		s.as_ref(),
		t.as_ref(),
		None,
		None,
		Some(ul.as_mut()),
		Some(ur.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
	// the homogeneous values come from the other public saved-pencil routine,
	// so its order and conjugate encoding are checked against this one
	let (mut ar, mut ai, mut beta) = (
		Col::<TR>::zeros(n),
		Col::<TR>::zeros(n),
		Col::<TR>::zeros(n),
	);
	schur_eigenvalues_real(
		s.as_ref(),
		t.as_ref(),
		ar.as_mut(),
		ai.as_mut(),
		beta.as_mut(),
	)
	.unwrap();
	let mut pairs: Vec<Pair> = Vec::new();
	let mut j = 0;
	while j < n {
		let nw = if j + 1 < n && s[(j + 1, j)] != zero() {
			2
		} else {
			1
		};
		pairs.push((j, nw, up(&ar[j]) + i() * up(&ai[j]), up(&beta[j])));
		j += nw;
	}
	assert_eq!(pairs.len(), 6);
	// a homogeneous pair is defined up to a nonzero real factor, so the finite
	// values are compared as `alpha / beta`
	let tol = 64.0 * eps;
	let want = [
		(1, Some(c64::new(2.0, 0.0))),
		(2, Some(c64::new(1.0, 1.0) / 1.5)),
		(1, None),
		(1, Some(c64::new(2.0, 0.0))),
		(1, None),
	];
	for (k, &(nw, lam)) in want.iter().enumerate() {
		assert_eq!(pairs[k].1, nw, "pair {k}: width");
		let (alpha, beta) = (pairs[k].2, pairs[k].3);
		match lam {
			None => assert!(
				beta.norm() == 0.0 && alpha.norm() > 0.0 && alpha.im == 0.0,
				"pair {k}: ({alpha:?}, {beta:?}) is not an infinite real \
				 eigenvalue"
			),
			Some(lam) => assert!(
				beta.norm() > 0.0 && close64(alpha / beta, lam, tol),
				"pair {k}: {:?} != {lam:?}",
				alpha / beta
			),
		}
	}
	assert!(
		pairs[5].2.norm() == 0.0 && pairs[5].3.norm() == 0.0,
		"pair 5: ({:?}, {:?}) is not the 0/0 block",
		pairs[5].2,
		pairs[5].3
	);
	check_pairs(
		s.as_ref(),
		t.as_ref(),
		ul.as_ref(),
		ur.as_ref(),
		&pairs,
		&up,
		10.0 * n as f64 * eps,
		true,
	);
}

/// 5x5 complex generalized Schur pencil: `1 + 2i` over `1`, `2` over `0.5`, an
/// infinite `3/0`, a nonregular `0/0`, and `(-2 + i)/2`, with complex coupling
/// above the diagonal. `T`'s diagonal is real, as `qz_cplx` leaves it (see the
/// rustdoc of [`gevd_from_schur_cplx`]).
fn saved_cplx_pencil<T: ComplexField>(
	lift: impl Fn(f64, f64) -> T,
) -> (Mat<T>, Mat<T>) {
	let n = 5;
	let mut s = Mat::<T>::zeros(n, n);
	let mut t = Mat::<T>::zeros(n, n);
	for j in 0..n {
		for r in 0..j {
			s[(r, j)] = lift(
				0.1 * ((r + 2 * j) % 5) as f64 - 0.2,
				0.05 * ((2 * r + j) % 3) as f64,
			);
			t[(r, j)] = lift(
				0.04 * ((3 * r + j) % 7) as f64 - 0.12,
				-0.03 * ((r + j) % 4) as f64,
			);
		}
	}
	s[(0, 0)] = lift(1.0, 2.0);
	t[(0, 0)] = lift(1.0, 0.0);
	s[(1, 1)] = lift(2.0, 0.0);
	t[(1, 1)] = lift(0.5, 0.0);
	s[(2, 2)] = lift(3.0, 0.0);
	s[(3, 3)] = lift(0.0, 0.0);
	s[(4, 4)] = lift(-2.0, 1.0);
	t[(4, 4)] = lift(2.0, 0.0);
	(s, t)
}

fn check_saved_cplx<TC: ComplexField>(
	lift: impl Fn(f64, f64) -> TC,
	up: impl Fn(&TC) -> c64,
	eps: f64,
) {
	let (s, t) = saved_cplx_pencil(lift);
	let n = s.nrows();
	let (mut ul, mut ur) = (Mat::<TC>::zeros(n, n), Mat::<TC>::zeros(n, n));
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<TC>(n, Par::Seq.shape()));
	gevd_from_schur_cplx(
		s.as_ref(),
		t.as_ref(),
		None,
		None,
		Some(ul.as_mut()),
		Some(ur.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
	// the homogeneous values of a triangular pencil are its diagonals
	let pairs: Vec<Pair> = (0..n)
		.map(|j| (j, 1, up(&s[(j, j)]), up(&t[(j, j)])))
		.collect();
	let tol = 64.0 * eps;
	assert!(close64(pairs[2].2, c64::new(3.0, 0.0), tol));
	assert!(pairs[2].3.norm() == 0.0, "column 2 must be infinite");
	assert!(
		pairs[3].2.norm() == 0.0 && pairs[3].3.norm() == 0.0,
		"column 3 must be the 0/0 block"
	);
	check_pairs(
		s.as_ref(),
		t.as_ref(),
		ul.as_ref(),
		ur.as_ref(),
		&pairs,
		&up,
		10.0 * n as f64 * eps,
		false,
	);
}

/// the issue's MWE: eigenvalues `2/1` and `3/0`, in that order
fn check_mwe<T: RealField>(tol: f64) {
	let s = Mat::<T>::from_fn(2, 2, |r, c| {
		from_f64([[2.0, 1.0], [0.0, 3.0]][r][c])
	});
	let t = Mat::<T>::from_fn(2, 2, |r, c| {
		from_f64([[1.0, 0.5], [0.0, 0.0]][r][c])
	});
	let mut u = Mat::<T>::zeros(2, 2);
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<T>(2, Par::Seq.shape()));
	gevd_from_schur_real(
		s.as_ref(),
		t.as_ref(),
		None,
		None,
		None,
		Some(u.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
	let tol = from_f64::<T>(tol);
	// eigenvalue 2/1: the null vector of `S - 2T` is e0
	assert!(abs(&(u[(0, 0)].abs() - &one::<T>())) <= tol);
	assert!(u[(1, 0)].abs() <= tol);
	// eigenvalue 3/0 (infinite): `T v = 0` is `v0 + 0.5 v1 = 0`
	assert!(abs(&(&u[(0, 1)] + &(&u[(1, 1)] * &from_f64::<T>(0.5)))) <= tol);
	assert!(u[(1, 1)].abs() > from_f64::<T>(0.5));
	// unit norms, in the order the pencil has them
	for j in 0..2 {
		let norm = sqrt::<T>(
			&(u[(0, j)].abs() * &u[(0, j)].abs()
				+ &u[(1, j)].abs() * &u[(1, j)].abs()),
		);
		assert!(abs(&(norm - one::<T>())) <= tol);
	}
}

#[test]
fn mwe_vectors_keep_the_diagonal_order() {
	check_mwe::<f32>(1e-6);
	check_mwe::<f64>(1e-14);
}

#[test]
fn saved_pencil_vectors_match_the_other_public_entry_point() {
	check_saved_real::<f32>(
		|x: &f32| c64::new(*x as f64, 0.0),
		f32::EPSILON as f64,
	);
	check_saved_real::<f64>(|x: &f64| c64::new(*x, 0.0), f64::EPSILON);
	check_saved_cplx::<c32>(
		|re, im| c32::new(re as f32, im as f32),
		|x: &c32| c64::new(x.re as f64, x.im as f64),
		f32::EPSILON as f64,
	);
	check_saved_cplx::<c64>(c64::new, |x: &c64| *x, f64::EPSILON);
}

struct RealSaved<T: RealField> {
	original_a: Mat<T>,
	original_b: Mat<T>,
	s: Mat<T>,
	t: Mat<T>,
	q: Mat<T>,
	z: Mat<T>,
	ar: Col<T>,
	ai: Col<T>,
	beta: Col<T>,
}

/// a real pencil and its public QZ: `s`, `t`, `q`, `z` and the homogeneous
/// values of the saved pencil, plus the pencil before the reduction
fn real_schur<T: RealField>(n: usize) -> RealSaved<T> {
	let mut a = Mat::<T>::from_fn(n, n, |r, c| from_f64(sample(r, c)));
	// `generalized_hessenberg` requires B triangular on input
	let mut b = Mat::<T>::from_fn(n, n, |r, c| {
		if r > c {
			zero()
		} else {
			from_f64(sample(r + 11, c + 5) + if r == c { 2.0 } else { 0.0 })
		}
	});
	let (original_a, original_b) = (a.clone(), b.clone());
	let (mut q, mut z) = (Mat::<T>::identity(n, n), Mat::<T>::identity(n, n));
	let (mut ar, mut ai, mut beta) =
		(Col::<T>::zeros(n), Col::<T>::zeros(n), Col::<T>::zeros(n));
	let hp: gevd::gen_hessenberg::GeneralizedHessenbergParams =
		colmat::auto!(T);
	let qp: gevd::GeneralizedSchurParams = colmat::auto!(T);
	let mut mem = MemBuffer::new(StackReq::all_of(&[
		gevd::gen_hessenberg::generalized_hessenberg_scratch::<T>(n, hp),
		gevd::qz_real::hessenberg_to_qz_scratch::<T>(n, ParShape::Seq, qp),
	]));
	gevd::gen_hessenberg::generalized_hessenberg(
		a.as_mut(),
		b.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		hp,
	);
	gevd::qz_real::hessenberg_to_qz(
		a.as_mut(),
		b.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		ar.as_mut(),
		ai.as_mut(),
		beta.as_mut(),
		ComputeEigenvectors::Yes,
		Par::Seq,
		qp,
		MemStack::new(&mut mem),
	);
	RealSaved {
		original_a,
		original_b,
		s: a,
		t: b,
		q,
		z,
		ar,
		ai,
		beta,
	}
}

struct CplxSaved<T: ComplexField> {
	original_a: Mat<T>,
	original_b: Mat<T>,
	s: Mat<T>,
	t: Mat<T>,
	q: Mat<T>,
	z: Mat<T>,
	alpha: Col<T>,
	beta: Col<T>,
}

/// the complex counterpart of [`real_schur`]
fn cplx_schur<TC: ComplexField>(
	n: usize,
	lift: impl Fn(f64, f64) -> TC,
) -> CplxSaved<TC> {
	let mut a = Mat::<TC>::from_fn(n, n, |r, c| {
		lift(sample(r, c), sample(r + 3, c + 7))
	});
	// `generalized_hessenberg` requires B triangular on input
	let mut b = Mat::<TC>::from_fn(n, n, |r, c| {
		if r > c {
			zero()
		} else {
			lift(
				sample(r + 11, c + 5) + if r == c { 2.0 } else { 0.0 },
				sample(r + 13, c + 2),
			)
		}
	});
	let (original_a, original_b) = (a.clone(), b.clone());
	let (mut q, mut z) = (Mat::<TC>::identity(n, n), Mat::<TC>::identity(n, n));
	let (mut alpha, mut beta) = (Col::<TC>::zeros(n), Col::<TC>::zeros(n));
	let hp: gevd::gen_hessenberg::GeneralizedHessenbergParams =
		colmat::auto!(TC);
	let qp: gevd::GeneralizedSchurParams = colmat::auto!(TC);
	let mut mem = MemBuffer::new(StackReq::all_of(&[
		gevd::gen_hessenberg::generalized_hessenberg_scratch::<TC>(n, hp),
		gevd::qz_cplx::hessenberg_to_qz_scratch::<TC>(n, ParShape::Seq, qp),
	]));
	gevd::gen_hessenberg::generalized_hessenberg(
		a.as_mut(),
		b.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		hp,
	);
	gevd::qz_cplx::hessenberg_to_qz(
		a.as_mut(),
		b.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		alpha.as_mut(),
		beta.as_mut(),
		ComputeEigenvectors::Yes,
		Par::Seq,
		qp,
		MemStack::new(&mut mem),
	);
	CplxSaved {
		original_a,
		original_b,
		s: a,
		t: b,
		q,
		z,
		alpha,
		beta,
	}
}

fn check_back_transform_real<TR: RealField>(up: impl Fn(&TR) -> c64, eps: f64) {
	let n = 8;
	let d = real_schur::<TR>(n);
	let tol = 10.0 * n as f64 * eps;
	let mut pairs: Vec<Pair> = Vec::new();
	let mut j = 0;
	while j < n {
		let nw = if j + 1 < n && d.s[(j + 1, j)] != zero() {
			2
		} else {
			1
		};
		pairs.push((j, nw, up(&d.ar[j]) + i() * up(&d.ai[j]), up(&d.beta[j])));
		j += nw;
	}
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<TR>(n, Par::Seq.shape()));
	let mut run = |a: MatRef<'_, TR>,
	               b: MatRef<'_, TR>,
	               q: Option<MatRef<'_, TR>>,
	               z: Option<MatRef<'_, TR>>,
	               par: Par| {
		let (mut ul, mut ur) = (Mat::<TR>::zeros(n, n), Mat::<TR>::zeros(n, n));
		gevd_from_schur_real(
			a,
			b,
			q,
			z,
			Some(ul.as_mut()),
			Some(ur.as_mut()),
			par,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		(ul, ur)
	};
	// (a) back-transformed: the vectors of the pencil before the reduction
	let (ul, ur) = run(
		d.s.as_ref(),
		d.t.as_ref(),
		Some(d.q.as_ref()),
		Some(d.z.as_ref()),
		Par::Seq,
	);
	check_pairs(
		d.original_a.as_ref(),
		d.original_b.as_ref(),
		ul.as_ref(),
		ur.as_ref(),
		&pairs,
		&up,
		tol,
		true,
	);
	// (b) without the transformation: the vectors of the saved pencil
	let (sl, sr) = run(d.s.as_ref(), d.t.as_ref(), None, None, Par::Seq);
	check_pairs(
		d.s.as_ref(),
		d.t.as_ref(),
		sl.as_ref(),
		sr.as_ref(),
		&pairs,
		&up,
		tol,
		true,
	);
	// (c) a team of two agrees within the family's tolerance; bitwise
	// agreement is promised at a fixed thread count only
	// (docs/reproducibility.md)
	let (tl, tr) = colmat::team::with_team(2, |par| {
		run(
			d.s.as_ref(),
			d.t.as_ref(),
			Some(d.q.as_ref()),
			Some(d.z.as_ref()),
			par,
		)
	});
	close(tl.as_ref(), ul.as_ref());
	close(tr.as_ref(), ur.as_ref());
}

fn check_back_transform_cplx<TC: ComplexField>(
	lift: impl Fn(f64, f64) -> TC,
	up: impl Fn(&TC) -> c64,
	eps: f64,
) {
	let n = 8;
	let d = cplx_schur(n, lift);
	let tol = 10.0 * n as f64 * eps;
	let pairs: Vec<Pair> = (0..n)
		.map(|j| (j, 1, up(&d.alpha[j]), up(&d.beta[j])))
		.collect();
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<TC>(n, Par::Seq.shape()));
	let mut run = |a: MatRef<'_, TC>,
	               b: MatRef<'_, TC>,
	               q: Option<MatRef<'_, TC>>,
	               z: Option<MatRef<'_, TC>>| {
		let (mut ul, mut ur) = (Mat::<TC>::zeros(n, n), Mat::<TC>::zeros(n, n));
		gevd_from_schur_cplx(
			a,
			b,
			q,
			z,
			Some(ul.as_mut()),
			Some(ur.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		(ul, ur)
	};
	let (ul, ur) = run(
		d.s.as_ref(),
		d.t.as_ref(),
		Some(d.q.as_ref()),
		Some(d.z.as_ref()),
	);
	check_pairs(
		d.original_a.as_ref(),
		d.original_b.as_ref(),
		ul.as_ref(),
		ur.as_ref(),
		&pairs,
		&up,
		tol,
		false,
	);
	let (sl, sr) = run(d.s.as_ref(), d.t.as_ref(), None, None);
	check_pairs(
		d.s.as_ref(),
		d.t.as_ref(),
		sl.as_ref(),
		sr.as_ref(),
		&pairs,
		&up,
		tol,
		false,
	);
}

#[test]
fn back_transform_recovers_the_original_pencil() {
	check_back_transform_real::<f32>(
		|x: &f32| c64::new(*x as f64, 0.0),
		f32::EPSILON as f64,
	);
	check_back_transform_real::<f64>(|x: &f64| c64::new(*x, 0.0), f64::EPSILON);
	check_back_transform_cplx::<c32>(
		|re, im| c32::new(re as f32, im as f32),
		|x: &c32| c64::new(x.re as f64, x.im as f64),
		f32::EPSILON as f64,
	);
	check_back_transform_cplx::<c64>(c64::new, |x: &c64| *x, f64::EPSILON);
}

#[test]
fn strided_inputs_and_reused_outputs() {
	let (sf, tf) = saved_real_pencil();
	let n = sf.nrows();
	// the same pencil as an offset view with a leading dimension larger than
	// `n`, and in row-major storage (row stride one, column stride `n`)
	let mut pad_s = Mat::<f64>::zeros(n + 3, n + 2);
	let mut pad_t = Mat::<f64>::zeros(n + 3, n + 2);
	pad_s
		.as_mut()
		.submatrix_mut(1, 1, n, n)
		.copy_from(sf.as_ref());
	pad_t
		.as_mut()
		.submatrix_mut(1, 1, n, n)
		.copy_from(tf.as_ref());
	let s_view = pad_s.as_ref().submatrix(1, 1, n, n);
	let t_view = pad_t.as_ref().submatrix(1, 1, n, n);
	let (mut s_rows, mut t_rows) = (vec![0.0f64; n * n], vec![0.0f64; n * n]);
	for r in 0..n {
		for c in 0..n {
			s_rows[r * n + c] = sf[(r, c)];
			t_rows[r * n + c] = tf[(r, c)];
		}
	}
	let s_row = MatRef::from_row_major_slice(&s_rows, n, n);
	let t_row = MatRef::from_row_major_slice(&t_rows, n, n);
	let run = |s: MatRef<'_, f64>, t: MatRef<'_, f64>, init: f64| {
		let mut ul = Mat::<f64>::from_fn(n, n, |_, _| init);
		let mut ur = Mat::<f64>::from_fn(n, n, |_, _| init);
		let mut mem =
			MemBuffer::new(gevd_from_schur_scratch::<f64>(n, Par::Seq.shape()));
		gevd_from_schur_real(
			s,
			t,
			None,
			None,
			Some(ul.as_mut()),
			Some(ur.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		(ul, ur)
	};
	let (ul, ur) = run(sf.as_ref(), tf.as_ref(), 0.0);
	let (ul_off, ur_off) = run(s_view, t_view, 0.0);
	close(ul.as_ref(), ul_off.as_ref());
	close(ur.as_ref(), ur_off.as_ref());
	let (ul_row, ur_row) = run(s_row, t_row, 0.0);
	close(ul.as_ref(), ul_row.as_ref());
	close(ur.as_ref(), ur_row.as_ref());
	// the outputs are outputs only: a buffer pre-filled with NaN gives the same
	// columns bit for bit
	let (ul_nan, ur_nan) = run(sf.as_ref(), tf.as_ref(), f64::NAN);
	assert_eq!(ul, ul_nan);
	assert_eq!(ur, ur_nan);
	// and the back-transformation takes strided `q` / `z`
	let d = real_schur::<f64>(n);
	let mut pad_q = Mat::<f64>::zeros(n + 2, n + 1);
	pad_q
		.as_mut()
		.submatrix_mut(1, 0, n, n)
		.copy_from(d.q.as_ref());
	let back = |q: MatRef<'_, f64>| {
		let mut ul = Mat::<f64>::zeros(n, n);
		let mut mem =
			MemBuffer::new(gevd_from_schur_scratch::<f64>(n, Par::Seq.shape()));
		gevd_from_schur_real(
			d.s.as_ref(),
			d.t.as_ref(),
			Some(q),
			Some(d.z.as_ref()),
			Some(ul.as_mut()),
			None,
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		ul
	};
	let ul_q = back(d.q.as_ref());
	let ul_q_stride = back(pad_q.as_ref().submatrix(1, 0, n, n));
	close(ul_q.as_ref(), ul_q_stride.as_ref());
}

#[test]
fn nonfinite_input_is_rejected() {
	let (sf, tf) = saved_real_pencil();
	let n = sf.nrows();
	let mut s = sf.clone();
	s[(2, 5)] = f64::NAN;
	let (mut ul, mut ur) = (Mat::<f64>::zeros(n, n), Mat::<f64>::zeros(n, n));
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<f64>(n, Par::Seq.shape()));
	let err = gevd_from_schur_real(
		s.as_ref(),
		tf.as_ref(),
		None,
		None,
		Some(ul.as_mut()),
		Some(ur.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	);
	assert_eq!(err, Err(gevd::GevdError::NoConvergence));
	for u in [ul.as_ref(), ur.as_ref()] {
		for r in 0..n {
			for c in 0..n {
				assert!(u[(r, c)].is_nan(), "a rejected output must be NaN");
			}
		}
	}
	// a non-finite transformation is rejected even when its side is not
	// requested
	let mut q = Mat::<f64>::identity(n, n);
	q[(0, 1)] = f64::INFINITY;
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<f64>(n, Par::Seq.shape()));
	let err = gevd_from_schur_real(
		sf.as_ref(),
		tf.as_ref(),
		Some(q.as_ref()),
		None,
		None,
		Some(ur.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	);
	assert_eq!(err, Err(gevd::GevdError::NoConvergence));
}

#[test]
fn empty_and_singleton_pencils() {
	// n = 0: nothing to write, and no vectors requested is not an error
	let s = Mat::<f64>::zeros(0, 0);
	let t = Mat::<f64>::zeros(0, 0);
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<f64>(0, Par::Seq.shape()));
	gevd_from_schur_real(
		s.as_ref(),
		t.as_ref(),
		None,
		None,
		None,
		None,
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
	// n = 1, finite and infinite, one vector per call
	for (sv, tv) in [(2.0f64, 1.0f64), (3.0, 0.0), (0.0, 0.0)] {
		let s = Mat::<f64>::from_fn(1, 1, |_, _| sv);
		let t = Mat::<f64>::from_fn(1, 1, |_, _| tv);
		let mut u = Mat::<f64>::zeros(1, 1);
		let mut mem =
			MemBuffer::new(gevd_from_schur_scratch::<f64>(1, Par::Seq.shape()));
		gevd_from_schur_real(
			s.as_ref(),
			t.as_ref(),
			None,
			None,
			Some(u.as_mut()),
			None,
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		)
		.unwrap();
		assert!((u[(0, 0)].abs() - 1.0).abs() < 1e-15);
	}
}

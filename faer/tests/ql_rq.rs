//! Public contract of the packed QL and RQ factorizations (colmat-rs#60): exact
//! scratch queries, every output mode into reused buffers, the trapezoidal shape
//! and the positive gauge of the triangular factor, rank-deficient and graded
//! inputs, a row-major input and strided outputs, `Par::Seq` against a two-thread
//! team, and the MatrixFactorizations.jl reference factors
//! (`fixtures/ql_rq_reference/`).
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::{ql, rq};
use colmat::traits::ComplexField;
use colmat::{Auto, Mat, Par, ParShape, c32, c64};
mod common;
use common::*;

fn sample_mat<T: ComplexField>(m: usize, n: usize, seed: f64) -> Mat<T> {
	Mat::from_fn(m, n, |i, j| {
		T::from_f64(sample(i, j) + seed * ((i + 2 * j) as f64).sin())
	})
}

fn block_size<T: ComplexField>(m: usize, n: usize) -> usize {
	let k = Ord::min(m, n);
	Ord::max(
		ql::no_pivoting::factor::recommended_block_size::<T>(m, n)
			.min(Ord::max(k, 1)),
		1,
	)
}

fn garbage<T: ComplexField>(m: usize, n: usize) -> Mat<T> {
	Mat::full(m, n, T::from_f64(-7.5))
}

fn assert_zero<T: ComplexField>(a: MatRef<'_, T>) {
	assert!(a.nrows() * a.ncols() == 0 || a.norm_l2() == zero());
}

/// The diagonal that ends in the last row and column is real and nonnegative.
fn assert_positive<T: ComplexField>(d: T) {
	assert!(d.real() >= zero());
	let bound = eps::<T::Real>()
		* from_f64::<T::Real>(4.0)
		* d.abs().fmax(one::<T::Real>());
	assert!(d.imag().abs() <= bound);
}

/// QL packed form and coefficients of `a`.
fn ql_pack<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	par: Par<'_>,
) -> (Mat<T>, Mat<T>) {
	use ql::no_pivoting::factor;
	let (m, n) = a.shape();
	let params = <factor::QlParams as Auto<T>>::auto().into();
	let mut mem = MemBuffer::new(factor::ql_in_place_scratch::<T>(
		m,
		n,
		block_size,
		par.shape(),
		params,
	));
	let mut packed = a.to_owned();
	let mut coeff =
		Mat::<T>::full(block_size, Ord::min(m, n), T::from_f64(f64::INFINITY));
	factor::ql_in_place(
		packed.as_mut(),
		coeff.as_mut(),
		par,
		MemStack::new(&mut mem),
		params,
	);
	(packed, coeff)
}

/// RQ packed form and coefficients of `a`.
fn rq_pack<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	par: Par<'_>,
) -> (Mat<T>, Mat<T>) {
	use rq::no_pivoting::factor;
	let (m, n) = a.shape();
	let params = <factor::RqParams as Auto<T>>::auto().into();
	let mut mem = MemBuffer::new(factor::rq_in_place_scratch::<T>(
		m,
		n,
		block_size,
		par.shape(),
		params,
	));
	let mut packed = a.to_owned();
	let mut coeff =
		Mat::<T>::full(block_size, Ord::min(m, n), T::from_f64(f64::INFINITY));
	factor::rq_in_place(
		packed.as_mut(),
		coeff.as_mut(),
		par,
		MemStack::new(&mut mem),
		params,
	);
	(packed, coeff)
}

/// Thin `Q` (`m x k`) and compact `L` (`k x n`) of `a` in the positive gauge.
fn ql_factors<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	mut par: Par<'_>,
) -> (Mat<T>, Mat<T>) {
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let (packed, coeff) = ql_pack(a, block_size, par.reborrow());
	let mut q = garbage::<T>(m, k);
	let mut l = garbage::<T>(k, n);
	let mut mem =
		MemBuffer::new(ql::no_pivoting::unitary::unitary_scratch::<T>(
			m,
			n,
			k,
			block_size,
			par.shape(),
		));
	ql::no_pivoting::unitary::unitary(
		q.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		par,
		MemStack::new(&mut mem),
	);
	ql::no_pivoting::factor::copy_l(l.as_mut(), packed.as_ref(), true);
	(q, l)
}

/// Compact `R` (`m x k`) and thin `Q` (`k x n`) of `a` in the positive gauge.
fn rq_factors<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	mut par: Par<'_>,
) -> (Mat<T>, Mat<T>) {
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let (packed, coeff) = rq_pack(a, block_size, par.reborrow());
	let mut r = garbage::<T>(m, k);
	let mut q = garbage::<T>(k, n);
	let mut mem =
		MemBuffer::new(rq::no_pivoting::unitary::unitary_scratch::<T>(
			m,
			n,
			k,
			block_size,
			par.shape(),
		));
	rq::no_pivoting::unitary::unitary(
		q.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		par,
		MemStack::new(&mut mem),
	);
	rq::no_pivoting::factor::copy_r(r.as_mut(), packed.as_ref(), true);
	(r, q)
}

/// Every QL output mode at the exact scratch size, into reused buffers.
fn ql_contract<T: ComplexField>(a: MatRef<'_, T>) {
	use ql::no_pivoting::{factor, reconstruct, unitary};
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let bs = block_size::<T>(m, n);
	let (packed, coeff) = ql_pack(a, bs, Par::Seq);
	let (q, l) = ql_factors(a, bs, Par::Seq);
	let mut l_full = garbage::<T>(m, n);
	let mut q_full = garbage::<T>(m, m);
	let mut rebuilt = garbage::<T>(m, n);
	factor::copy_l(l_full.as_mut(), packed.as_ref(), true);
	unitary::unitary(
		q_full.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(unitary::unitary_scratch::<T>(
			m,
			n,
			m,
			bs,
			ParShape::Seq,
		))),
	);
	reconstruct::reconstruct(
		rebuilt.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(
			reconstruct::reconstruct_scratch::<T>(m, n, bs, ParShape::Seq),
		)),
	);

	// $A = QL$, thin and full, and the packed rebuild.
	close((&q * &l).as_ref(), a);
	close((&q_full * &l_full).as_ref(), a);
	close(rebuilt.as_ref(), a);
	unitary(q.as_ref());
	unitary(q_full.as_ref());
	// the thin factors are the trailing columns of the full Q and the trailing
	// rows of the full L, whose leading rows are zero.
	close(q_full.as_ref().get(.., m - k..), q.as_ref());
	close(l_full.as_ref().get(m - k.., ..), l.as_ref());
	assert_zero(l_full.as_ref().get(..m - k, ..));
	// lower trapezoidal, ending in the last row and column, with a positive
	// diagonal.
	for r in 0..k {
		assert_zero(l.as_ref().get(r..r + 1, r + n - k + 1..));
		assert_positive(l[(r, r + n - k)].clone());
	}
}

/// Every RQ output mode at the exact scratch size, into reused buffers.
fn rq_contract<T: ComplexField>(a: MatRef<'_, T>) {
	use rq::no_pivoting::{factor, reconstruct, unitary};
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let bs = block_size::<T>(m, n);
	let (packed, coeff) = rq_pack(a, bs, Par::Seq);
	let (r, q) = rq_factors(a, bs, Par::Seq);
	let mut r_full = garbage::<T>(m, n);
	let mut q_full = garbage::<T>(n, n);
	let mut rebuilt = garbage::<T>(m, n);
	factor::copy_r(r_full.as_mut(), packed.as_ref(), true);
	unitary::unitary(
		q_full.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(unitary::unitary_scratch::<T>(
			m,
			n,
			n,
			bs,
			ParShape::Seq,
		))),
	);
	reconstruct::reconstruct(
		rebuilt.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(
			reconstruct::reconstruct_scratch::<T>(m, n, bs, ParShape::Seq),
		)),
	);

	// $A = RQ$, thin and full, and the packed rebuild.
	close((&r * &q).as_ref(), a);
	close((&r_full * &q_full).as_ref(), a);
	close(rebuilt.as_ref(), a);
	close(
		(&q * q.adjoint()).as_ref(),
		Mat::<T>::identity(k, k).as_ref(),
	);
	close(
		(&q_full * q_full.adjoint()).as_ref(),
		Mat::<T>::identity(n, n).as_ref(),
	);
	// the thin factors are the trailing rows of the full Q and the trailing
	// columns of the full R, whose leading columns are zero.
	close(q_full.as_ref().get(n - k.., ..), q.as_ref());
	close(r_full.as_ref().get(.., n - k..), r.as_ref());
	assert_zero(r_full.as_ref().get(.., ..n - k));
	// upper trapezoidal, ending in the last row and column, with a positive
	// diagonal.
	for c in 0..k {
		assert_zero(r.as_ref().get(c + m - k + 1.., c..c + 1));
		assert_positive(r[(c + m - k, c)].clone());
	}
}

const SHAPES: [(usize, usize); 9] = [
	(0, 0),
	(0, 3),
	(3, 0),
	(1, 1),
	(8, 8),
	(8, 3),
	(3, 8),
	(100, 33),
	(33, 100),
];

fn contracts<T: ComplexField>() {
	for (m, n) in SHAPES {
		let a = sample_mat::<T>(m, n, 0.0);
		ql_contract(a.as_ref());
		rq_contract(a.as_ref());
	}
	// rank deficient: a repeated column and a zero row
	let mut a = sample_mat::<T>(9, 6, 0.5);
	for i in 0..9 {
		a[(i, 4)] = a[(i, 1)].clone();
		if i < 6 {
			a[(3, i)] = zero();
		}
	}
	for b in [a.as_ref(), a.transpose()] {
		ql_contract(b);
		rq_contract(b);
	}
	// graded rows, twelve orders of magnitude apart
	let graded = Mat::<T>::from_fn(12, 7, |i, j| {
		T::from_f64(sample(i, j) * 10f64.powi(-(i as i32)))
	});
	for b in [graded.as_ref(), graded.transpose()] {
		ql_contract(b);
		rq_contract(b);
	}
}

#[test]
fn ql_rq_contract() {
	contracts::<f64>();
	contracts::<f32>();
	contracts::<c64>();
	contracts::<c32>();
}

/// A row-major input and strided outputs give the column-major factors.
fn views<T: ComplexField>(m: usize, n: usize) {
	let k = Ord::min(m, n);
	let bs = block_size::<T>(m, n);
	let a = sample_mat::<T>(m, n, 0.25);
	let row_major = a.transpose().to_owned();
	let (q, l) = ql_factors(a.as_ref(), bs, Par::Seq);
	let (q_rm, l_rm) = ql_factors(row_major.transpose(), bs, Par::Seq);
	close(q_rm.as_ref(), q.as_ref());
	close(l_rm.as_ref(), l.as_ref());
	let (r, qr) = rq_factors(a.as_ref(), bs, Par::Seq);
	let (r_rm, qr_rm) = rq_factors(row_major.transpose(), bs, Par::Seq);
	close(r_rm.as_ref(), r.as_ref());
	close(qr_rm.as_ref(), qr.as_ref());

	// strided outputs inside a guarded buffer
	let (packed, coeff) = ql_pack(a.as_ref(), bs, Par::Seq);
	let mut wide = Mat::<T>::full(m + 3, k + 2, T::from_f64(2.5));
	ql::no_pivoting::unitary::unitary(
		wide.as_mut().submatrix_mut(1, 2, m, k),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(
			ql::no_pivoting::unitary::unitary_scratch::<T>(
				m,
				n,
				k,
				bs,
				ParShape::Seq,
			),
		)),
	);
	close(wide.as_ref().submatrix(1, 2, m, k), q.as_ref());
	assert_eq!(wide[(0, 0)], T::from_f64(2.5));
	assert_eq!(wide[(m + 2, k + 1)], T::from_f64(2.5));
}

#[test]
fn ql_rq_views() {
	for (m, n) in [(8, 3), (3, 8), (8, 8), (100, 33)] {
		views::<f64>(m, n);
		views::<c64>(m, n);
	}
}

#[cfg(feature = "threadteam")]
#[test]
fn ql_rq_parallel_matches_sequential() {
	fn agree<T: ComplexField>(m: usize, n: usize) {
		let k = Ord::min(m, n);
		// shifted on the diagonal that ends in the last row and column, so the
		// condition number is small and Q is compared entry by entry
		let mut a = sample_mat::<T>(m, n, 0.5);
		for r in 0..k {
			let (i, j) = (r + m - k, r + n - k);
			a[(i, j)] = a[(i, j)].clone() + T::from_f64(16.0);
		}
		let bs = Ord::max(block_size::<T>(m, n) / 2, 1);
		let (q_seq, l_seq) = ql_factors(a.as_ref(), bs, Par::Seq);
		let (r_seq, qr_seq) = rq_factors(a.as_ref(), bs, Par::Seq);
		colmat::team::with_team(2, |mut par: Par<'_>| {
			let (q, l) = ql_factors(a.as_ref(), bs, par.reborrow());
			close(q.as_ref(), q_seq.as_ref());
			close(l.as_ref(), l_seq.as_ref());
			let (r, qr) = rq_factors(a.as_ref(), bs, par);
			close(r.as_ref(), r_seq.as_ref());
			close(qr.as_ref(), qr_seq.as_ref());
		});
	}
	for (m, n) in [(257, 129), (129, 257), (257, 257)] {
		agree::<f64>(m, n);
		agree::<c64>(m, n);
	}
}

/// One case of `fixtures/ql_rq_reference/ql_rq.txt`.
struct Case {
	kind: String,
	complex: bool,
	mats: Vec<(String, Mat<f64>)>,
}

fn parse(text: &str) -> Vec<Case> {
	let mut cases: Vec<Case> = Vec::new();
	let mut lines = text.lines().filter(|l| !l.starts_with('#'));
	while let Some(line) = lines.next() {
		let words: Vec<&str> = line.split_whitespace().collect();
		if words[0] == "case" {
			cases.push(Case {
				kind: words[1].to_string(),
				complex: words[2] == "c64",
				mats: Vec::new(),
			});
			continue;
		}
		let (rows, cols): (usize, usize) =
			(words[1].parse().unwrap(), words[2].parse().unwrap());
		let mut mat = Mat::<f64>::zeros(rows, cols);
		for i in 0..rows {
			let values: Vec<f64> = lines
				.next()
				.unwrap()
				.split_whitespace()
				.map(|v| v.parse().unwrap())
				.collect();
			assert_eq!(values.len(), cols);
			for (j, v) in values.into_iter().enumerate() {
				mat[(i, j)] = v;
			}
		}
		cases
			.last_mut()
			.unwrap()
			.mats
			.push((words[0].to_string(), mat));
	}
	cases
}

/// The named matrix of `case`, its entries built by `scalar(re, im)`.
fn get<T: ComplexField>(
	case: &Case,
	name: &str,
	scalar: &impl Fn(f64, f64) -> T,
) -> Mat<T> {
	let find = |key: String| {
		case.mats
			.iter()
			.find(|(n, _)| *n == key)
			.map(|(_, m)| m.clone())
			.unwrap()
	};
	let (re, im) = if case.complex {
		(find(format!("{name}_re")), find(format!("{name}_im")))
	} else {
		let re = find(name.to_string());
		let im = Mat::zeros(re.nrows(), re.ncols());
		(re, im)
	};
	Mat::from_fn(re.nrows(), re.ncols(), |i, j| {
		scalar(re[(i, j)], im[(i, j)])
	})
}

/// With the positive gauge a full-rank factorization is unique, so both factors
/// match MatrixFactorizations.jl entry by entry.
fn matches_reference<T: ComplexField>(
	case: &Case,
	scalar: impl Fn(f64, f64) -> T,
) {
	let a = get(case, "a", &scalar);
	let (q_ref, t_ref) = (get(case, "q", &scalar), get(case, "t", &scalar));
	let bs = block_size::<T>(a.nrows(), a.ncols());
	if case.kind == "ql" {
		let (q, l) = ql_factors(a.as_ref(), bs, Par::Seq);
		close(q.as_ref(), q_ref.as_ref());
		close(l.as_ref(), t_ref.as_ref());
	} else {
		let (r, q) = rq_factors(a.as_ref(), bs, Par::Seq);
		close(q.as_ref(), q_ref.as_ref());
		close(r.as_ref(), t_ref.as_ref());
	}
}

#[test]
fn ql_rq_match_matrixfactorizations_jl() {
	let cases = parse(include_str!("fixtures/ql_rq_reference/ql_rq.txt"));
	assert_eq!(cases.len(), 12);
	for case in &cases {
		if case.complex {
			matches_reference(case, c64::new);
		} else {
			matches_reference(case, |re, _| re);
		}
	}
}

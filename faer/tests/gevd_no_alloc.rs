//! tensor4all/faer#375: with the scratch sized by `gevd_scratch`, a warm call of
//! `gevd_real` / `gevd_cplx` allocates nothing on the heap, also when `A` and
//! `B` are prescaled (entries far below `sqrt(safmin) / eps` or above its
//! inverse) and `alpha`, `beta` are scaled back.
//!
//! The allocator counts the allocations of the calling thread while the call
//! runs (a thread-local flag, so tests running on other threads are not
//! counted); the inputs, the outputs and the scratch are built outside the
//! counted interval. `Par::Seq`, so the call does not leave the thread.
#![cfg(feature = "std")]
use colmat::diag::Diag;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::evd::ComputeEigenvectors;
use colmat::linalg::gevd::{
	GevdParams, gevd_cplx, gevd_from_schur_cplx, gevd_from_schur_real,
	gevd_from_schur_scratch, gevd_real, gevd_scratch,
};
use colmat::{Mat, Par, ParShape, Spec, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
	static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
fn count() {
	// `try_with`: the allocator also runs while thread locals are torn down
	let _ = COUNTING.try_with(|c| {
		if c.get() {
			let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
		}
	});
}
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		count();
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		count();
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		count();
		unsafe { System.realloc(ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations of the second of two calls of `run`: first-use allocations
/// belong to the warm-up.
fn warm(mut run: impl FnMut()) -> usize {
	run();
	ALLOCATIONS.with(|a| a.set(0));
	COUNTING.with(|c| c.set(true));
	run();
	COUNTING.with(|c| c.set(false));
	ALLOCATIONS.with(|a| a.get())
}

/// A fixed pseudo-random value in `[-1, 1)`.
fn sample(i: usize, j: usize, seed: u64) -> f64 {
	let mut s = (i as u64 + 1)
		.wrapping_mul(0x9e37_79b9_7f4a_7c15)
		.wrapping_add((j as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9))
		.wrapping_add(seed.wrapping_mul(0x94d0_49bb_1331_11eb));
	s ^= s >> 31;
	s = s.wrapping_mul(0xd6e8_feb8_6659_fd93);
	s ^= s >> 32;
	((s >> 11) as f64 / (1u64 << 52) as f64) - 1.0
}

/// `1` is in range; `1e-300` and `1e300` are prescaled
const SCALES: [f64; 3] = [1.0, 1e-300, 1e300];
/// below and above the QZ's default blocking threshold (75)
const SIZES: [usize; 2] = [20, 100];

fn real_allocations(n: usize, s: f64, ce: ComputeEigenvectors) -> usize {
	let a = Mat::<f64>::from_fn(n, n, |i, j| s * sample(i, j, 1));
	let b = Mat::<f64>::from_fn(n, n, |i, j| {
		s * (sample(i, j, 2) + if i == j { 2.0 } else { 0.0 })
	});
	let (mut am, mut bm) = (a.clone(), b.clone());
	let mut re = Diag::<f64>::zeros(n);
	let mut im = Diag::<f64>::zeros(n);
	let mut beta = Diag::<f64>::zeros(n);
	let mut ul = Mat::<f64>::zeros(n, n);
	let mut ur = Mat::<f64>::zeros(n, n);
	let params = Spec::<GevdParams, f64>::default();
	let mut mem =
		MemBuffer::new(gevd_scratch::<f64>(n, ce, ce, ParShape::Seq, params));
	let vectors = matches!(ce, ComputeEigenvectors::Yes);
	warm(|| {
		am.copy_from(&a);
		bm.copy_from(&b);
		gevd_real(
			am.as_mut(),
			bm.as_mut(),
			re.as_mut(),
			im.as_mut(),
			beta.as_mut(),
			vectors.then(|| ul.as_mut()),
			vectors.then(|| ur.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap();
	})
}

fn cplx_allocations(n: usize, s: f64, ce: ComputeEigenvectors) -> usize {
	let a = Mat::<c64>::from_fn(n, n, |i, j| {
		c64::new(s * sample(i, j, 3), s * sample(i, j, 4))
	});
	let b = Mat::<c64>::from_fn(n, n, |i, j| {
		c64::new(
			s * (sample(i, j, 5) + if i == j { 2.0 } else { 0.0 }),
			s * sample(i, j, 6),
		)
	});
	let (mut am, mut bm) = (a.clone(), b.clone());
	let mut alpha = Diag::<c64>::zeros(n);
	let mut beta = Diag::<c64>::zeros(n);
	let mut ul = Mat::<c64>::zeros(n, n);
	let mut ur = Mat::<c64>::zeros(n, n);
	let params = Spec::<GevdParams, c64>::default();
	let mut mem =
		MemBuffer::new(gevd_scratch::<c64>(n, ce, ce, ParShape::Seq, params));
	let vectors = matches!(ce, ComputeEigenvectors::Yes);
	warm(|| {
		am.copy_from(&a);
		bm.copy_from(&b);
		gevd_cplx(
			am.as_mut(),
			bm.as_mut(),
			alpha.as_mut(),
			beta.as_mut(),
			vectors.then(|| ul.as_mut()),
			vectors.then(|| ur.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap();
	})
}

#[test]
fn gevd_warm_call_does_not_allocate() {
	let mut failures = Vec::new();
	for n in SIZES {
		for s in SCALES {
			for ce in [ComputeEigenvectors::No, ComputeEigenvectors::Yes] {
				let r = real_allocations(n, s, ce);
				let c = cplx_allocations(n, s, ce);
				println!(
					"n = {n}, s = {s:e}, vectors = {ce:?}: gevd_real {r}, gevd_cplx {c} allocations"
				);
				if r != 0 || c != 0 {
					failures.push((n, s, ce, r, c));
				}
			}
		}
	}
	assert!(failures.is_empty(), "allocations: {failures:?}");
}

/// A saved real Schur pencil: quasi-triangular `S`, triangular `T`, a `2x2`
/// block with equal diagonals at the top, and infinite eigenvalues at the end.
fn saved_real(n: usize) -> (Mat<f64>, Mat<f64>) {
	let mut s = Mat::<f64>::zeros(n, n);
	let mut t = Mat::<f64>::zeros(n, n);
	for j in 0..n {
		for r in 0..j {
			s[(r, j)] = 0.05 * sample(r, j, 7);
			t[(r, j)] = 0.05 * sample(r, j, 8);
		}
		s[(j, j)] = 2.0 + j as f64 * 0.25;
		t[(j, j)] = 1.0 + j as f64 * 0.125;
	}
	s[(0, 0)] = 1.5;
	s[(0, 1)] = 0.75;
	s[(1, 0)] = -0.5;
	s[(1, 1)] = 1.5;
	t[(0, 0)] = 1.0;
	t[(1, 1)] = 1.0;
	for j in n.saturating_sub(2)..n {
		t[(j, j)] = 0.0;
	}
	(s, t)
}

/// The complex counterpart, with `T`'s diagonal real, as `qz_cplx` leaves it.
fn saved_cplx(n: usize) -> (Mat<c64>, Mat<c64>) {
	let mut s = Mat::<c64>::zeros(n, n);
	let mut t = Mat::<c64>::zeros(n, n);
	for j in 0..n {
		for r in 0..j {
			s[(r, j)] =
				c64::new(0.05 * sample(r, j, 7), 0.05 * sample(r, j, 9));
			t[(r, j)] =
				c64::new(0.05 * sample(r, j, 8), 0.05 * sample(r, j, 10));
		}
		s[(j, j)] = c64::new(2.0 + j as f64 * 0.25, 0.5);
		t[(j, j)] = c64::new(1.0 + j as f64 * 0.125, 0.0);
	}
	for j in n.saturating_sub(2)..n {
		t[(j, j)] = c64::new(0.0, 0.0);
	}
	(s, t)
}

fn from_schur_real_allocations(n: usize) -> usize {
	let (s, t) = saved_real(n);
	let (q, z) = (Mat::<f64>::identity(n, n), Mat::<f64>::identity(n, n));
	let (mut ul, mut ur) = (Mat::<f64>::zeros(n, n), Mat::<f64>::zeros(n, n));
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<f64>(n, ParShape::Seq));
	warm(|| {
		gevd_from_schur_real(
			s.as_ref(),
			t.as_ref(),
			Some(q.as_ref()),
			Some(z.as_ref()),
			Some(ul.as_mut()),
			Some(ur.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			Spec::default(),
		)
		.unwrap();
	})
}

fn from_schur_cplx_allocations(n: usize) -> usize {
	let (s, t) = saved_cplx(n);
	let (q, z) = (Mat::<c64>::identity(n, n), Mat::<c64>::identity(n, n));
	let (mut ul, mut ur) = (Mat::<c64>::zeros(n, n), Mat::<c64>::zeros(n, n));
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<c64>(n, ParShape::Seq));
	warm(|| {
		gevd_from_schur_cplx(
			s.as_ref(),
			t.as_ref(),
			Some(q.as_ref()),
			Some(z.as_ref()),
			Some(ul.as_mut()),
			Some(ur.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			Spec::default(),
		)
		.unwrap();
	})
}

#[test]
fn gevd_from_schur_warm_call_does_not_allocate() {
	let mut failures = Vec::new();
	for n in [8usize, 100] {
		let r = from_schur_real_allocations(n);
		let c = from_schur_cplx_allocations(n);
		println!(
			"n = {n}: gevd_from_schur_real {r}, gevd_from_schur_cplx {c} \
			 allocations"
		);
		if r != 0 || c != 0 {
			failures.push((n, r, c));
		}
	}
	assert!(failures.is_empty(), "allocations: {failures:?}");
}

//! #169: a warm `matmul` on the `gemm`-crate route must not allocate.
//!
//! The `gemm`-crate route is what `matmul` takes on aarch64 and on x86_64 without AVX2+FMA;
//! on x86_64 with AVX2+FMA or AVX-512 every native scalar goes to `private_gemm_x86`, which
//! keeps its own thread-local buffer and allocates nothing. This test therefore only checks
//! anything on the former hosts, and returns early on the latter.
//!
//! Blocked upstream: `t4a-gemm-common` keeps its per-thread packing buffer only up to
//! `CACHE_INFO[1].cache_bytes` (`PackBuffer`'s `Drop`, gemm-common/src/gemm.rs), so a product
//! whose packed right-hand side is larger than L2 allocates once per call. Measured on
//! tensor4all/t4a-gemm main 61d7b4b (x86_64, calling `gemm::gemm` directly with a row-major
//! right-hand side): 256^3 f64 one allocation of 528 KB per warm call, 1024^3 f64 one of
//! 4.2 MB. `gemm::gemm` takes no caller buffer, so colmat cannot avoid it. The test is ignored
//! until the dependency keeps a larger buffer or offers a caller-scratch entry point.
#![cfg(feature = "std")]
use colmat::{Mat, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
			BYTES.fetch_add(layout.size(), Ordering::Relaxed);
		}
		unsafe { System.alloc(layout) }
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
			BYTES.fetch_add(layout.size(), Ordering::Relaxed);
		}
		unsafe { System.alloc_zeroed(layout) }
	}
	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
			BYTES.fetch_add(new_size, Ordering::Relaxed);
		}
		unsafe { System.realloc(ptr, layout, new_size) }
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Whether `matmul` takes the `private_gemm_x86` route on this host, where this test checks
/// nothing.
fn x86_kernel_route() -> bool {
	#[cfg(target_arch = "x86_64")]
	{
		std::arch::is_x86_feature_detected!("avx512f")
			|| (std::arch::is_x86_feature_detected!("avx2")
				&& std::arch::is_x86_feature_detected!("fma"))
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		false
	}
}

fn sample(m: usize, n: usize, seed: u64) -> Mat<f64> {
	let mut x = seed
		.wrapping_mul(6364136223846793005)
		.wrapping_add(1442695040888963407);
	Mat::from_fn(m, n, |_, _| {
		x = x
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
	})
}

// one test function: the counter is process wide, so nothing else may run while it counts
#[test]
#[ignore = "blocked on t4a-gemm-common: the per-thread pack buffer is kept only up to L2 size (colmat#169)"]
fn warm_matmul_on_the_gemm_crate_route_allocates_nothing() {
	if x86_kernel_route() {
		eprintln!(
			"skipped: matmul takes private_gemm_x86 on this host, not the gemm-crate route"
		);
		return;
	}
	// Square products whose packed right-hand side is below and above L2, with a row-major
	// right-hand side so that the route packs it on every target.
	for n in [48usize, 256, 1024] {
		let a = sample(n, n, 1);
		let b_storage = sample(n, n, 2);
		let b = b_storage.transpose();
		let mut c = Mat::<f64>::zeros(n, n);
		let mut call = || {
			colmat::linalg::matmul::matmul(
				c.as_mut(),
				colmat::Accum::Replace,
				a.as_ref(),
				b,
				1.0,
				Par::Seq,
			)
		};
		call();
		call(); // warm-up: thread locals and the kept pack buffer
		ALLOCATIONS.store(0, Ordering::SeqCst);
		BYTES.store(0, Ordering::SeqCst);
		COUNTING.store(true, Ordering::SeqCst);
		call();
		COUNTING.store(false, Ordering::SeqCst);
		let (count, bytes) = (
			ALLOCATIONS.load(Ordering::SeqCst),
			BYTES.load(Ordering::SeqCst),
		);
		assert_eq!(
			count, 0,
			"{n}x{n}x{n} f64, Par::Seq: {count} allocations ({bytes} bytes) in a warm call"
		);
	}
}

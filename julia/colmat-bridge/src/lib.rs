//! Private benchmark bridge, not colmat's supported C ABI. Owning decompositions allocate
//! factors/workspace inside the call; output destruction/copy and pool startup are outside timing.
use colmat::linalg::solvers::{Qr, SelfAdjointEigen, Svd};
use colmat::{Mat, MatRef, Par, Side};
use std::panic::{AssertUnwindSafe, catch_unwind};

struct TeamState {
	team: Option<threadteam::Team>,
	width: usize,
}
enum Output {
	Eigh(SelfAdjointEigen<f64>),
	Qr { factor: Qr<f64>, q: Mat<f64> },
    RawQr { _packed: Mat<f64>, _coeff: Mat<f64>, q: Mat<f64>, r: Mat<f64>, times: [f64;2] },
	Svd(Svd<f64>),
}

#[no_mangle]
pub extern "C" fn compare_team_new(width: usize) -> *mut std::ffi::c_void {
	catch_unwind(|| {
		assert!((1..=16).contains(&width));
		let team = if width == 1 {
			None
		} else {
			let pool = threadteam::Pool::builder()
				.threads(width)
				.acquire_timeout(std::time::Duration::from_secs(30))
				.build()
				.expect("comparison pool");
			Some(pool.into_team(width).ok().expect("idle comparison pool"))
		};
		Box::into_raw(Box::new(TeamState { team, width })).cast()
	})
	.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `handle` is a live handle returned by compare_team_new, used exclusively by this caller;
/// input is a readable nonaliasing column-major array of m*n finite f64s, alive until return.
/// kind=0 requires a square symmetric matrix. This private driver does not accept empty shapes.
#[no_mangle]
pub unsafe extern "C" fn compare_run(
	handle: *mut std::ffi::c_void,
	input: *const f64,
	m: usize,
	n: usize,
	kind: u32,
) -> *mut std::ffi::c_void {
	catch_unwind(AssertUnwindSafe(|| {
		assert!(!handle.is_null() && !input.is_null() && m > 0 && n > 0);
		let len = m.checked_mul(n).expect("matrix size");
		assert!(len <= isize::MAX as usize / size_of::<f64>());
		// SAFETY: the private Julia driver lends its live dense Matrix for the synchronous call.
		let data = unsafe { std::slice::from_raw_parts(input, len) };
		let a = MatRef::from_column_major_slice(data, m, n);
		// SAFETY: the handle is exclusively borrowed by the caller and not freed during this call.
		let state = unsafe { &mut *handle.cast::<TeamState>() };
		let operation = |mut par: Par<'_>| match kind {
			0 => {
				assert_eq!(m, n);
				Output::Eigh(
					a.self_adjoint_eigen(Side::Lower, par).expect("EVD"),
				)
			},
			1 | 3 => {
				let factor = a.qr(par.reborrow());
                // Same factor code and held team; kind 3 reconstructs the original
                // compute_thin_Q with the public arbitrary-input sequence API.
                let q = if kind == 1 { factor.compute_thin_Q(par) } else {
                    use colmat::dyn_stack::{MemBuffer, MemStack};
                    use colmat::linalg::householder::{apply_block_householder_sequence_on_the_left_in_place_scratch, apply_block_householder_sequence_on_the_left_in_place_with_conj};
                    let k = m.min(n);
                    let mut q = Mat::identity(m,k);
                    let mut mem = MemBuffer::new(apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(m,factor.Q_coeff().nrows(),k));
                    apply_block_householder_sequence_on_the_left_in_place_with_conj(factor.Q_basis(),factor.Q_coeff(),colmat::Conj::No,q.as_mut(),par,MemStack::new(&mut mem));
                    q
                };
				Output::Qr { factor, q }
			},
			2 => Output::Svd(a.thin_svd(par).expect("SVD")),
            100..=164 => {
                use colmat::dyn_stack::{MemBuffer,MemStack};
                use colmat::linalg::qr::no_pivoting::{factor,unitary};
                let k=m.min(n);let b=48.min(k).max(1);
                let mut packed=a.to_owned();let mut h=Mat::zeros(b,k);
                let params=colmat::Spec::new(factor::QrParams { recursion_leaf_size:(kind-100) as usize, ..<factor::QrParams as colmat::Auto<f64>>::auto() });
                let mut mem=MemBuffer::new(factor::qr_in_place_scratch::<f64>(m,n,b,par.shape(),params));
                let t=std::time::Instant::now();
                factor::qr_in_place(packed.as_mut(),h.as_mut(),par.reborrow(),MemStack::new(&mut mem),params);
                let tf=t.elapsed().as_secs_f64();
                let mut q=Mat::zeros(m,k);
                let mut mem=MemBuffer::new(unitary::unitary_scratch::<f64>(m,n,k,b,par.shape()));
                let t=std::time::Instant::now();
                unitary::unitary(q.as_mut(),packed.as_ref(),h.as_ref(),false,par.reborrow(),MemStack::new(&mut mem));
                let tq=t.elapsed().as_secs_f64();
                let mut r=Mat::zeros(k,n);factor::copy_r(r.as_mut(),packed.as_ref(),false);
                Output::RawQr{_packed:packed,_coeff:h,q,r,times:[tf,tq]}
            },
			_ => panic!("unknown decomposition"),
		};
		let result = if let Some(team) = &state.team {
			team.exec(state.width)
				.team(|t| operation(Par::Team(t)))
				.ok()
				.expect("reserved team")
		} else {
			operation(Par::Seq)
		};
		Box::into_raw(Box::new(result)).cast()
	}))
	.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// handle is a live compare_run result. dst is disjoint writable storage with exactly the
/// selected factor's number of elements. The Julia driver supplies those dimensions.
/// Eigh: selectors0=values,1=U; QR:1=Q,2=thin R; SVD:0=values,1=U,2=V.
#[no_mangle]
pub unsafe extern "C" fn compare_copy(
	handle: *const std::ffi::c_void,
	which: u32,
	dst: *mut f64,
) {
	// SAFETY: caller supplies a live Output and matching disjoint destination.
	let result = unsafe { &*handle.cast::<Output>() };
	let write_mat = |a: MatRef<'_, f64>| {
		for j in 0..a.ncols() {
			for i in 0..a.nrows() {
				// SAFETY: output length is the selected matrix's element count, column-major.
				unsafe {
					*dst.add(i + j * a.nrows()) = a[(i, j)];
				}
			}
		}
	};
	match (result, which) {
		(Output::Eigh(f), 0) => {
			for i in 0..f.S().dim() {
				// SAFETY: output holds the n eigenvalues.
				unsafe {
					*dst.add(i) = f.S()[i];
				}
			}
		},
		(Output::Svd(f), 0) => {
			for i in 0..f.S().dim() {
				// SAFETY: output holds min(m,n) singular values.
				unsafe {
					*dst.add(i) = f.S()[i];
				}
			}
		},
        (Output::RawQr { q,.. },1)=>write_mat(q.as_ref()),
        (Output::RawQr { r,.. },2)=>write_mat(r.as_ref()),
        (Output::RawQr { times,.. },0)=> {
            // SAFETY: the private leaf probe supplies a writable two-element timing array.
            for i in 0..2 { unsafe { *dst.add(i)=times[i]; } }
        },
		(Output::Eigh(f), 1) => write_mat(f.U()),
		(Output::Qr { q, .. }, 1) => write_mat(q.as_ref()),
		(Output::Qr { factor, .. }, 2) => write_mat(factor.thin_R()),
		(Output::Svd(f), 1) => write_mat(f.U()),
		(Output::Svd(f), 2) => write_mat(f.V()),
		_ => (),
	}
}

/// # Safety
/// handle is a live result, released exactly once after all reads have ended.
#[no_mangle]
pub unsafe extern "C" fn compare_result_free(handle: *mut std::ffi::c_void) {
	if !handle.is_null() {
		// SAFETY: exclusive ownership of the Box is returned by the private driver.
		drop(unsafe { Box::from_raw(handle.cast::<Output>()) });
	}
}
/// # Safety
/// handle is a live team handle, freed once, with no active call using it.
#[no_mangle]
pub unsafe extern "C" fn compare_team_free(handle: *mut std::ffi::c_void) {
	if !handle.is_null() {
		// SAFETY: exclusive ownership of the Box is returned by the private driver.
		drop(unsafe { Box::from_raw(handle.cast::<TeamState>()) });
	}
}

/// Diagnostic stage times for owning QR and thin-Q construction.
/// # Safety
/// Same live team/input requirements as compare_run. times points to three writable f64s.
#[no_mangle]
pub unsafe extern "C" fn compare_qr_profile(
	handle: *mut std::ffi::c_void,
	input: *const f64,
	m: usize,
	n: usize,
	times: *mut f64,
) -> i32 {
	let result = catch_unwind(AssertUnwindSafe(|| {
		// SAFETY: the private Julia driver supplies a live handle and matrix for this call.
		let state = unsafe { &mut *handle.cast::<TeamState>() };
		// SAFETY: matrix dimensions and storage are checked by the private driver.
		let a = MatRef::from_column_major_slice(
			unsafe { std::slice::from_raw_parts(input, m * n) },
			m,
			n,
		);
		let operation = |mut par: Par<'_>| {
			let t = std::time::Instant::now();
			let factor = a.qr(par.reborrow());
			let factor_time = t.elapsed().as_secs_f64();
			let t = std::time::Instant::now();
			let q = factor.compute_thin_Q(par);
			let basis_time = t.elapsed().as_secs_f64();
			std::hint::black_box((&factor, &q));
			[factor_time, basis_time, factor_time + basis_time]
		};
		let t = if let Some(team) = &state.team {
			team.exec(state.width)
				.team(|p| operation(Par::Team(p)))
				.ok()
				.expect("reserved team")
		} else {
			operation(Par::Seq)
		};
		for i in 0..3 {
			// SAFETY: the driver lends three writable output elements.
			unsafe {
				*times.add(i) = t[i];
			}
		}
	}));
	if result.is_ok() { 0 } else { -1 }
}

use super::LINEAR_IMPL_THRESHOLD;
use crate::internal_prelude::*;
use colmat_traits::RealReg;
use num_complex::Complex;

#[inline]
fn range_scales<R: RealField>() -> (R, R) {
	// Widen both power-of-two scales so even MIN_SUBNORMAL and MAX can be squared.
	let eps = &eps::<R>();
	(
		sqrt_min_positive::<R>() * eps,
		sqrt_max_positive::<R>() / eps,
	)
}

#[inline(always)]
fn norm_l2_simd<'N, T: ComplexField>(
	data: ColRef<'_, T, Dim<'N>, ContiguousFwd>,
) -> [T::Real; 3] {
	struct Impl<'a, 'N, T: ComplexField> {
		data: ColRef<'a, T, Dim<'N>, ContiguousFwd>,
	}
	impl<'N, T: ComplexField> pulp::WithSimd for Impl<'_, 'N, T> {
		type Output = [T::Real; 3];

		#[inline(always)]
		fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
			let Self { data } = self;
			let simd = SimdCtx::<T, S>::new(T::simd_ctx(simd), data.nrows());
			let (down, up) = range_scales::<T::Real>();
			let sml = simd.splat_real(down);
			let big = simd.splat_real(up);
			let mut acc_sml = [RealReg(simd.zero()); 2];
			let mut acc_med = [RealReg(simd.zero()); 2];
			let mut acc_big = [RealReg(simd.zero()); 2];

			simd_iter!(for (IDX, i) in [simd.batch_indices::<2>(); 2] {
				let x = simd.read(data, i);
				acc_sml[IDX] =
					simd.abs2_add(simd.mul_real(x, sml), acc_sml[IDX]);
				acc_med[IDX] = simd.abs2_add(x, acc_med[IDX]);
				acc_big[IDX] =
					simd.abs2_add(simd.mul_real(x, big), acc_big[IDX]);
			});

			let acc0_sml = RealReg(simd.add(acc_sml[0].0, acc_sml[1].0));
			let acc0_big = RealReg(simd.add(acc_big[0].0, acc_big[1].0));
			let acc0_med = RealReg(simd.add(acc_med[0].0, acc_med[1].0));
			[
				simd.reduce_sum_real(acc0_sml),
				simd.reduce_sum_real(acc0_med),
				simd.reduce_sum_real(acc0_big),
			]
		}
	}
	dispatch!(Impl { data }, Impl, T)
}
fn norm_l2_simd_pairwise_rows<T: ComplexField>(
	data: ColRef<'_, T, usize, ContiguousFwd>,
) -> [T::Real; 3] {
	if data.nrows() <= LINEAR_IMPL_THRESHOLD {
		with_dim!(N, data.nrows());
		norm_l2_simd(data.as_row_shape(N))
	} else {
		let split_point = data.nrows().div_ceil(2).next_power_of_two();
		let (head, tail) = data.split_at_row(split_point);
		let acc0 = norm_l2_simd_pairwise_rows(head);
		let acc1 = norm_l2_simd_pairwise_rows(tail);
		[
			&acc0[0] + &acc1[0],
			&acc0[1] + &acc1[1],
			&acc0[2] + &acc1[2],
		]
	}
}
fn norm_l2_simd_pairwise_cols<T: ComplexField>(
	data: MatRef<'_, T, usize, usize, ContiguousFwd>,
) -> [T::Real; 3] {
	if data.ncols() == 1 {
		norm_l2_simd_pairwise_rows(data.col(0))
	} else {
		let split_point = data.ncols().div_ceil(2).next_power_of_two();
		let (head, tail) = data.split_at_col(split_point);
		let acc0 = norm_l2_simd_pairwise_cols(head);
		let acc1 = norm_l2_simd_pairwise_cols(tail);
		[
			&acc0[0] + &acc1[0],
			&acc0[1] + &acc1[1],
			&acc0[2] + &acc1[2],
		]
	}
}
fn norm_l2_x3<T: ComplexField>(mut mat: MatRef<'_, T>) -> [T::Real; 3] {
	if mat.ncols() > 1 && mat.col_stride().unsigned_abs() == 1 {
		mat = mat.transpose();
	}
	if mat.row_stride() < 0 {
		mat = mat.reverse_rows();
	}
	if mat.nrows() == 0 || mat.ncols() == 0 {
		[zero(), zero(), zero()]
	} else {
		let m = mat.nrows();
		let n = mat.ncols();
		if const { T::SIMD_CAPABILITIES.is_simd() } {
			if let Some(mat) = mat.try_as_col_major() {
				if const { T::IS_NATIVE_C32 } {
					let mat: MatRef<
						'_,
						Complex<f32>,
						usize,
						usize,
						ContiguousFwd,
					> = unsafe { crate::hacks::coerce(mat) };
					let mat = unsafe {
						MatRef::<'_, f32, usize, usize, ContiguousFwd>::from_raw_parts(
							mat.as_ptr() as *const f32,
							2 * mat.nrows(),
							mat.ncols(),
							ContiguousFwd,
							mat.col_stride().wrapping_mul(2),
						)
					};
					return unsafe {
						crate::hacks::coerce(norm_l2_simd_pairwise_cols::<f32>(
							mat,
						))
					};
				} else if const { T::IS_NATIVE_C64 } {
					let mat: MatRef<
						'_,
						Complex<f64>,
						usize,
						usize,
						ContiguousFwd,
					> = unsafe { crate::hacks::coerce(mat) };
					let mat = unsafe {
						MatRef::<'_, f64, usize, usize, ContiguousFwd>::from_raw_parts(
							mat.as_ptr() as *const f64,
							2 * mat.nrows(),
							mat.ncols(),
							ContiguousFwd,
							mat.col_stride().wrapping_mul(2),
						)
					};
					return unsafe {
						crate::hacks::coerce(norm_l2_simd_pairwise_cols::<f64>(
							mat,
						))
					};
				} else {
					return norm_l2_simd_pairwise_cols(mat);
				}
			}
		}
		let (down, up) = range_scales::<T::Real>();
		let (mut acc_sml, mut acc_med, mut acc_big) = (zero(), zero(), zero());
		// A running hypot can round away contributions at subnormal scale.
		for j in 0..n {
			for i in 0..m {
				let x = &mat[(i, j)];
				acc_sml = x.mul_pow2(&down).abs2() + acc_sml;
				acc_med = x.abs2() + acc_med;
				acc_big = x.mul_pow2(&up).abs2() + acc_big;
			}
		}
		[acc_sml, acc_med, acc_big]
	}
}
pub fn norm_l2<T: ComplexField>(mat: MatRef<'_, T>) -> T::Real {
	let [acc_sml, acc_med, acc_big] = norm_l2_x3(mat);
	let (down, up) = range_scales::<T::Real>();
	// Select while the unscaled sum is still far from the square's range limits.
	if acc_med < sqrt_min_positive() {
		acc_big.sqrt() * down
	} else if acc_med < sqrt_max_positive() {
		acc_med.sqrt()
	} else {
		// The scaled-down sum also propagates NaN, which compares false above.
		acc_sml.sqrt() * up
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Col, Mat, assert, c32, c64};

	#[test]
	fn test_norm_l2() {
		let relative_err =
			|a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());
		for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
			for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
				let mat = Mat::from_fn(m, n, |i, j| factor * ((i + j) as f64));
				let mut target = 0.0;
				zip!(mat.as_ref()).for_each(|unzip!(x)| {
					target = f64::hypot(*x, target);
				});
				if factor == 0.0 {
					assert!(norm_l2(mat.as_ref()) == target);
				} else {
					assert!(
						relative_err(norm_l2(mat.as_ref()), target) < 1e-14
					);
				}
			}
		}
		let mat = Col::from_fn(10000000, |_| 0.3);
		let target = (0.3 * 0.3 * 10000000.0f64).sqrt();
		assert!(relative_err(norm_l2(mat.as_ref().as_mat()), target) < 1e-14);
	}
	#[test]
	fn test_norm_l2_strided_range() {
		// Check unequal magnitudes on the generic-stride path.
		fn check<T: ComplexField>(scales: [f64; 3], tol: f64) {
			for s in scales {
				let mat = Mat::<T>::from_fn(2, 2, |i, j| match (i, j) {
					(0, 1) => from_f64::<T>(s * 0.5),
					(1, 0) => zero(),
					_ => from_f64::<T>(s),
				});
				let expected = from_f64::<T::Real>(s * 1.25f64.sqrt());
				let norm = mat.row(0).norm_l2();
				assert!(norm.is_finite());
				assert!(
					(norm - &expected).abs()
						<= expected.abs() * from_f64::<T::Real>(tol)
				);
			}
		}
		let wide = [1.0, 1e160, 1e-300];
		let narrow = [1.0, 1e30, 1e-35];
		check::<f32>(narrow, 1e-5);
		check::<c32>(narrow, 1e-5);
		check::<f64>(wide, 1e-14);
		check::<c64>(wide, 1e-14);
	}
	#[test]
	fn test_norm_l2_extreme_range() {
		fn check<T: ComplexField>(
			make: impl Fn(f64) -> T,
			max: f64,
			min_subnormal: f64,
			sqrt_max_positive: f64,
		) {
			for (n, x, expected) in [
				(1, 0.5 * max, 0.5 * max),
				(1, 0.75 * max, 0.75 * max),
				(4, 0.375 * max, 0.75 * max),
				(4, sqrt_max_positive, 2.0 * sqrt_max_positive),
				(1, min_subnormal, min_subnormal),
				(16, min_subnormal, 4.0 * min_subnormal),
				(16, 10_000.0 * min_subnormal, 40_000.0 * min_subnormal),
				(256, min_subnormal, 16.0 * min_subnormal),
				(4, 0.75 * max, f64::INFINITY),
				(4, f64::INFINITY, f64::INFINITY),
			] {
				let contiguous = Col::from_fn(n, |_| make(x));
				let strided = Mat::from_fn(2, n, |i, _| {
					if i == 0 { make(x) } else { make(f64::NAN) }
				});
				let expected = from_f64::<T::Real>(expected);
				for (layout, got) in [
					("contiguous", contiguous.norm_l2()),
					("strided", strided.row(0).norm_l2()),
				] {
					assert_eq!(got, expected, "{layout}, {n} copies of {x:e}");
				}
			}

			// NaN takes precedence over infinity, including in the imaginary part.
			let a = Mat::from_fn(2, 2, |i, j| {
				make(if i == j { f64::NAN } else { f64::INFINITY })
			});
			assert!(a.norm_l2().is_nan());
			assert!(a.row(0).norm_l2().is_nan());
		}

		let max32 = f32::MAX as f64;
		let min32 = f32::from_bits(1) as f64;
		let bound32 = 2.0f64.powi(63);
		check(|x| x as f32, max32, min32, bound32);
		check(|x| c32::new(0.0, x as f32), max32, min32, bound32);
		let bound64 = 2.0f64.powi(511);
		check(|x| x, f64::MAX, f64::from_bits(1), bound64);
		check(|x| c64::new(0.0, x), f64::MAX, f64::from_bits(1), bound64);
	}
	#[test]
	fn test_norm_l2_cplx() {
		let relative_err =
			|a: f64, b: f64| (a - b).abs() / f64::max(a.abs(), b.abs());
		for (m, n) in [(9, 10), (1023, 5), (42, 1)] {
			for factor in [0.0, 1.0, 1e30, 1e250, 1e-30, 1e-250] {
				let mat = Mat::from_fn(m, n, |i, j| {
					factor * c64::new((i + j) as f64, i.wrapping_sub(j) as f64)
				});
				let mut target = 0.0;
				zip!(mat.as_ref()).for_each(|unzip!(x)| {
					target = f64::hypot(x.abs(), target);
				});
				if factor == 0.0 {
					assert!(norm_l2(mat.as_ref()) == target);
				} else {
					assert!(
						relative_err(norm_l2(mat.as_ref()), target) < 1e-14
					);
				}
			}
		}
	}
}

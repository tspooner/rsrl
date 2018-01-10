//! Normalisation utilities module.

/// Computes the L1 norm.
#[inline]
pub fn l1(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |acc, v| acc + v.abs())
}

/// Computes the L2 norm.
#[inline]
pub fn l2(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |acc, v| acc + (v*v).abs()).sqrt()
}

/// Computes the Lp norm.
#[inline]
pub fn lp(x: &[f64], p: u8) -> f64 {
    x.iter().fold(0.0, |acc, v| acc + v.abs().powi(p as i32)).powf((p as f64).recip())
}

/// Computes the infinity norm.
#[inline]
pub fn linf(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |max, v| {
        let av = v.abs();

        if av > max {
            av
        } else {
            max
        }
    })
}

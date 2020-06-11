#![allow(dead_code)]
use ndarray::Array2;
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::f64;

pub fn argmaxima<I: IntoIterator<Item = f64>>(vals: I) -> (Vec<usize>, f64) {
    let mut max = f64::MIN;
    let mut ixs = vec![];

    for (i, v) in vals.into_iter().enumerate() {
        if (v - max).abs() < 1e-7 {
            ixs.push(i);
        } else if v > max {
            max = v;
            ixs.clear();
            ixs.push(i);
        }
    }

    (ixs, max)
}

pub fn argmax_first<I: IntoIterator<Item = f64>>(vals: I) -> (usize, f64) {
    vals.into_iter().enumerate().fold(
        (0, f64::MIN),
        |(i, x), (j, y)| {
            if y - x > 1e-7 {
                (j, y)
            } else {
                (i, x)
            }
        },
    )
}

pub fn argmax_last<I: IntoIterator<Item = f64>>(vals: I) -> (usize, f64) {
    vals.into_iter().enumerate().fold(
        (0, f64::MIN),
        |(i, x), (j, y)| {
            if y - x > -1e-7 {
                (j, y)
            } else {
                (i, x)
            }
        },
    )
}

pub fn argmax_choose<I: IntoIterator<Item = f64>>(vals: I) -> (usize, f64) {
    let (maxima, value) = argmaxima(vals);

    let maximum = if maxima.len() == 1 {
        maxima[0]
    } else {
        *maxima
            .choose(&mut thread_rng())
            .expect("No valid maxima to choose from in `argmax_choose`.")
    };

    (maximum, value)
}

pub fn argmax_choose_rng<R, I>(rng: &mut R, vals: I) -> (usize, f64)
where
    R: Rng + ?Sized,
    I: IntoIterator<Item = f64>,
{
    let (maxima, value) = argmaxima(vals);

    let maximum = if maxima.len() == 1 {
        maxima[0]
    } else {
        *maxima
            .choose(rng)
            .expect("No valid maxima to choose from in `argmax_choose`.")
    };

    (maximum, value)
}

/// Compute the pseudo-inverse of a real matrix using SVD.
pub fn pinv(m: &Array2<f64>) -> Result<Array2<f64>, ndarray_linalg::error::LinalgError> {
    use ndarray::Axis;
    use ndarray_linalg::svd::SVD;

    let m_dim = m.dim();
    let max_dim = m_dim.0.max(m_dim.1);

    m.svd(true, true).map(|(u, s, vt)| {
        // u: (M x M)
        // s: diag{(M x N)} => (max{M, N} x 1)
        // vt: (N x N)
        let u = u.unwrap();
        let vt = vt.unwrap();

        let threshold = f64::EPSILON
            * max_dim as f64
            * s.fold(
                unsafe { *s.uget(0) },
                |acc, &v| {
                    if v > acc {
                        v
                    } else {
                        acc
                    }
                },
            );

        // (max{M, N} x 1)
        let sinv = s
            .mapv(|v| if v > threshold { 1.0 / v } else { 0.0 })
            .insert_axis(Axis(1));

        vt.t().dot(&(&u.t() * &sinv))
    })
}

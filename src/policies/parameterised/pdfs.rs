use std::f64;

const PI_TIMES_2: f64 = 2.0 * f64::consts::PI;

pub(crate) fn normal_pdf(mu: f64, sigma: f64, x: f64) -> f64 {
    let diff = x - mu;

    let z = PI_TIMES_2.sqrt() * sigma;
    let e = -diff * diff / (2.0 * sigma * sigma);

    e.exp() / z
}

use ndarray::Array1;

pub(crate) fn runge_kutta4(fx: &Fn(f64, Array1<f64>) -> Array1<f64>,
                           x: f64, y: Array1<f64>, dx: f64) -> Array1<f64> {
    let k1 = dx*fx(x, y.clone());
    let k2 = dx*fx(x + dx/2.0, y.clone() + k1.clone()/2.0);
    let k3 = dx*fx(x + dx/2.0, y.clone() + k2.clone()/2.0);
    let k4 = dx*fx(x + dx, y.clone() + k3.clone());

    y + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
}

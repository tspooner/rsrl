pub(crate) fn runge_kutta4(fx: impl Fn(f64, Vec<f64>) -> Vec<f64>, x: f64, mut y: Vec<f64>, dx: f64) -> Vec<f64> {
    // Compute first term:
    let k1: Vec<f64> = fx(x, y.clone()).into_iter().map(|v| v * dx).collect();

    // Compute second term:
    let k2: Vec<f64> = fx(
        x + dx / 2.0,
        y.iter().zip(k1.iter()).map(|(a, b)| a + b / 2.0).collect()
    ).into_iter().map(|v| v * dx).collect();

    // Compute third term:
    let k3: Vec<f64> = fx(
        x + dx / 2.0,
        y.iter().zip(k2.iter()).map(|(a, b)| a + b / 2.0).collect()
    ).into_iter().map(|v| v * dx).collect();

    // Compute fourth term:
    let k4: Vec<f64> = fx(
        x + dx,
        y.iter().zip(k3.iter()).map(|(a, b)| a + b).collect()
    ).into_iter().map(|v| v * dx).collect();

    for i in 0..y.len() {
        y[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
    }

    y
}

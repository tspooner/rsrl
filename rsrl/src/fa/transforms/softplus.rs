use ndarray::Array1;
use super::{Transform, Logistic};

// f(x) ≜ log(1 + exp(x))
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Softplus;

impl Transform<f64> for Softplus {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        if x >= 10.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    fn grad(&self, x: f64) -> f64 {
        if x >= 10.0 {
            1.0
        } else {
            Logistic::sigmoid_stable(x)
        }
    }

    fn grad_scaled(&self, x: f64, error: f64) -> f64 {
        self.grad(x) * error
    }
}

impl_variants!(Softplus);

// f(x, y, ...) ≜ log(C + exp(x) + exp(y) + ...)
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct LogSumExp(f64);

impl LogSumExp {
    pub fn new(offset: f64) -> LogSumExp { LogSumExp(offset) }
}

impl Default for LogSumExp {
    fn default() -> LogSumExp { LogSumExp::new(0.0) }
}

impl Transform<f64> for LogSumExp {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        (self.0 + x.exp()).ln()
    }

    fn grad(&self, x: f64) -> f64 {
        let exp_term = x.exp();

        exp_term / (self.0 + exp_term)
    }

    fn grad_scaled(&self, x: f64, error: f64) -> f64 {
        self.grad(x) * error
    }
}

impl Transform<[f64; 2]> for LogSumExp {
    type Output = f64;

    fn transform(&self, x: [f64; 2]) -> f64 {
        (self.0 + x[0].exp() + x[1].exp()).ln()
    }

    fn grad(&self, x: [f64; 2]) -> [f64; 2] {
        let e = [x[0].exp(), x[1].exp()];
        let z = self.0 + e[0] + e[1];

        [e[0] / z, e[1] / z]
    }

    fn grad_scaled(&self, x: [f64; 2], error: f64) -> [f64; 2] {
        let e = [x[0].exp(), x[1].exp()];
        let z = self.0 + e[0] + e[1];

        [e[0] * error / z, e[1] * error / z]
    }
}

impl Transform<[f64; 3]> for LogSumExp {
    type Output = f64;

    fn transform(&self, x: [f64; 3]) -> f64 {
        (self.0 + x[0].exp() + x[1].exp() + x[2].exp()).ln()
    }

    fn grad(&self, x: [f64; 3]) -> [f64; 3] {
        let e = [x[0].exp(), x[1].exp(), x[2].exp()];
        let z = self.0 + e[0] + e[1] + e[2];

        [e[0] / z, e[1] / z, e[2] / z]
    }

    fn grad_scaled(&self, x: [f64; 3], error: f64) -> [f64; 3] {
        let e = [x[0].exp(), x[1].exp(), x[2].exp()];
        let z = self.0 + e[0] + e[1] + e[2];

        [e[0] * error / z, e[1] * error / z, e[2] * error / z]
    }
}

impl Transform<Array1<f64>> for LogSumExp {
    type Output = f64;

    fn transform(&self, x: Array1<f64>) -> f64 {
        (self.0 + x.into_iter().fold(0.0f64, |acc, v| acc + v.exp())).ln()
    }

    fn grad(&self, x: Array1<f64>) -> Array1<f64> {
        let e = x.mapv_into(|v| v.exp());
        let z = self.0 + e.sum();

        e / z
    }

    fn grad_scaled(&self, x: Array1<f64>, error: f64) -> Array1<f64> {
        let e = x.mapv_into(|v| v.exp());
        let z = self.0 + e.sum();

        e * (error / z)
    }
}

#[cfg(test)]
mod tests {
    use crate::fa::transforms::Logistic;
    use quickcheck::quickcheck;
    use super::{Softplus, Transform};

    #[test]
    fn test_f64() {
        assert!((Softplus.transform(0.0) - 0.693147).abs() < 1e-5);
        assert!((Softplus.transform(1.0) - 1.31326).abs() < 1e-5);
        assert!((Softplus.transform(2.0) - 2.12693).abs() < 1e-5);

        assert!((Softplus.grad(0.0) - Logistic::default().transform(0.0)).abs() < 1e-7);
        assert!((Softplus.grad(1.0) - Logistic::default().transform(1.0)).abs() < 1e-7);
        assert!((Softplus.grad(2.0) - Logistic::default().transform(2.0)).abs() < 1e-7);
    }

    #[test]
    fn test_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Softplus.transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}

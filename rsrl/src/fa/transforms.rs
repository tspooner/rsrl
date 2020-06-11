use ndarray::Array1;

/// An interface for differentiable transformations.
pub trait Transform<T: ?Sized> {
    type Output;

    /// Return the value of the transform for input `x`.
    fn transform(&self, x: T) -> Self::Output;

    /// Return the gradient of the transform for input `x`.
    fn grad(&self, x: T) -> T;

    /// Return the gradient of the transform for input `x` scaled the output
    /// error.
    fn grad_scaled(&self, x: T, error: Self::Output) -> T;
}

macro_rules! impl_variants {
    ($name:ident) => {
        impl Transform<[f64; 2]> for $name {
            type Output = [f64; 2];

            fn transform(&self, x: [f64; 2]) -> [f64; 2] {
                [
                    Transform::<f64>::transform(self, x[0]),
                    Transform::<f64>::transform(self, x[1]),
                ]
            }

            fn grad(&self, x: [f64; 2]) -> [f64; 2] {
                [
                    Transform::<f64>::grad(self, x[0]),
                    Transform::<f64>::grad(self, x[1]),
                ]
            }

            fn grad_scaled(&self, x: [f64; 2], errors: [f64; 2]) -> [f64; 2] {
                [
                    Transform::<f64>::grad(self, x[0]) * errors[0],
                    Transform::<f64>::grad(self, x[1]) * errors[1],
                ]
            }
        }

        impl Transform<[f64; 3]> for $name {
            type Output = [f64; 3];

            fn transform(&self, x: [f64; 3]) -> [f64; 3] {
                [
                    Transform::<f64>::transform(self, x[0]),
                    Transform::<f64>::transform(self, x[1]),
                    Transform::<f64>::transform(self, x[2]),
                ]
            }

            fn grad(&self, x: [f64; 3]) -> [f64; 3] {
                [
                    Transform::<f64>::grad(self, x[0]),
                    Transform::<f64>::grad(self, x[1]),
                    Transform::<f64>::grad(self, x[2]),
                ]
            }

            fn grad_scaled(&self, x: [f64; 3], errors: [f64; 3]) -> [f64; 3] {
                [
                    Transform::<f64>::grad(self, x[0]) * errors[0],
                    Transform::<f64>::grad(self, x[1]) * errors[1],
                    Transform::<f64>::grad(self, x[2]) * errors[2],
                ]
            }
        }

        impl Transform<Vec<f64>> for $name {
            type Output = Vec<f64>;

            fn transform(&self, x: Vec<f64>) -> Vec<f64> {
                x.into_iter()
                    .map(|v| Transform::<f64>::transform(self, v))
                    .collect()
            }

            fn grad(&self, x: Vec<f64>) -> Vec<f64> {
                x.into_iter()
                    .map(|v| Transform::<f64>::grad(self, v))
                    .collect()
            }

            fn grad_scaled(&self, x: Vec<f64>, errors: Vec<f64>) -> Vec<f64> {
                x.into_iter()
                    .zip(errors.into_iter())
                    .map(|(v, e)| Transform::<f64>::grad(self, v) * e)
                    .collect()
            }
        }

        impl Transform<Array1<f64>> for $name {
            type Output = Array1<f64>;

            fn transform(&self, x: Array1<f64>) -> Array1<f64> {
                x.mapv_into(|v| Transform::<f64>::transform(self, v))
            }

            fn grad(&self, x: Array1<f64>) -> Array1<f64> {
                x.mapv_into(|v| Transform::<f64>::grad(self, v))
            }

            fn grad_scaled(&self, x: Array1<f64>, errors: Array1<f64>) -> Array1<f64> {
                self.grad(x) * errors
            }
        }
    };
}

// f(x) ≜ x
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Identity;

macro_rules! impl_identity {
    ($type:ty; $grad:expr) => {
        impl Transform<$type> for Identity {
            type Output = $type;

            fn transform(&self, x: $type) -> $type { x }

            fn grad(&self, _: $type) -> $type { $grad }

            fn grad_scaled(&self, _: $type, error: $type) -> $type { error }
        }
    }
}

impl_identity!(f64; 1.0);
impl_identity!([f64; 2]; [1.0; 2]);
impl_identity!([f64; 3]; [1.0; 3]);

impl Transform<Vec<f64>> for Identity {
    type Output = Vec<f64>;

    fn transform(&self, x: Vec<f64>) -> Vec<f64> { x }

    fn grad(&self, x: Vec<f64>) -> Vec<f64> { x.into_iter().map(|_| 1.0).collect() }

    fn grad_scaled(&self, _: Vec<f64>, errors: Vec<f64>) -> Vec<f64> { errors }
}

impl Transform<Array1<f64>> for Identity {
    type Output = Array1<f64>;

    fn transform(&self, x: Array1<f64>) -> Array1<f64> { x }

    fn grad(&self, mut x: Array1<f64>) -> Array1<f64> { x.fill(1.0); x }

    fn grad_scaled(&self, _: Array1<f64>, errors: Array1<f64>) -> Array1<f64> { errors }
}

// f(x) ≜ tanh(x)
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Tanh;

impl Transform<f64> for Tanh {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn grad(&self, x: f64) -> f64 {
        let cosh = x.cosh();

        1.0 / cosh / cosh
    }

    fn grad_scaled(&self, x: f64, error: f64) -> f64 {
        self.grad(x) * error
    }
}

impl_variants!(Tanh);

// f(x) ≜ log(1 + exp(x))
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
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
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
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

// f(x) ≜ L / (1 + exp(-k(x - x0))
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Logistic {
    amplitude: f64,
    growth_rate: f64,
    midpoint: f64,
}

impl Logistic {
    pub fn new(amplitude: f64, growth_rate: f64, midpoint: f64) -> Logistic {
        Logistic { amplitude, growth_rate, midpoint, }
    }

    pub fn standard() -> Logistic {
        Logistic::new(1.0, 1.0, 0.0)
    }

    pub fn standard_scaled(amplitude: f64) -> Logistic {
        Logistic::new(amplitude, 1.0, 0.0)
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn sigmoid_stable(x: f64) -> f64 {
        if x >= 0.0 {
            Logistic::sigmoid(x)
        } else {
            let exp_x = x.exp();

            exp_x / (1.0 + exp_x)
        }
    }

    fn rescale_x(&self, x: f64) -> f64 { self.growth_rate * (x - self.midpoint) }
}

impl Default for Logistic {
    fn default() -> Logistic { Logistic::standard() }
}

impl Transform<f64> for Logistic {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        let x = self.rescale_x(x);

        self.amplitude * Logistic::sigmoid_stable(x)
    }

    fn grad(&self, x: f64) -> f64 {
        let x = self.rescale_x(x);
        let s = Logistic::sigmoid_stable(x);

        self.growth_rate * self.amplitude * (-x).exp() * s * s
    }

    fn grad_scaled(&self, x: f64, error: f64) -> f64 {
        self.grad(x) * error
    }
}

impl_variants!(Logistic);

// f(x) ≜ exp(x)
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Exp;

impl Transform<f64> for Exp {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        x.exp()
    }

    fn grad(&self, x: f64) -> f64 {
        self.transform(x)
    }

    fn grad_scaled(&self, x: f64, error: f64) -> f64 {
        self.grad(x) * error
    }
}

impl_variants!(Exp);

// TODO: Add implementations of standard activation functions.

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use std::f64::consts::E;
    use super::*;

    #[test]
    fn test_identity_scalar() {
        fn prop_transform(val: f64) -> bool {
            (Identity.transform(val) - val).abs() < 1e-7
        }

        fn prop_grad(val: f64) -> bool {
            (Identity.grad(val) - 1.0).abs() < 1e-7
        }

        quickcheck(prop_transform as fn(f64) -> bool);
        quickcheck(prop_grad as fn(f64) -> bool);
    }

    #[test]
    fn test_identity_pair() {
        fn prop_transform(val: (f64, f64)) -> bool {
            let t = Identity.transform([val.0, val.1]);

            (t[0] - val.0).abs() < 1e-7 && (t[1] - val.1).abs() < 1e-7
        }

        fn prop_grad(val: (f64, f64)) -> bool {
            let g = Identity.grad([val.0, val.1]);

            (g[0] - 1.0).abs() < 1e-7 && (g[1] - 1.0).abs() < 1e-7
        }

        quickcheck(prop_transform as fn((f64, f64)) -> bool);
        quickcheck(prop_grad as fn((f64, f64)) -> bool);
    }

    #[test]
    fn test_identity_triple() {
        fn prop_transform(val: (f64, f64, f64)) -> bool {
            let t = Identity.transform([val.0, val.1, val.2]);

            (t[0] - val.0).abs() < 1e-7 && (t[1] - val.1).abs() < 1e-7 && (t[2] - val.2).abs() < 1e-7
        }

        fn prop_grad(val: (f64, f64, f64)) -> bool {
            let g = Identity.grad([val.0, val.1, val.2]);

            (g[0] - 1.0).abs() < 1e-7 && (g[1] - 1.0).abs() < 1e-7 && (g[2] - 1.0).abs() < 1e-7
        }

        quickcheck(prop_transform as fn((f64, f64, f64)) -> bool);
        quickcheck(prop_grad as fn((f64, f64, f64)) -> bool);
    }

    #[test]
    fn test_identity_vector() {
        fn prop_transform(val: Vec<f64>) -> bool {
            let t = Identity.transform(Array1::from(val.clone()));

            t.into_iter().zip(val.into_iter()).all(|(v1, v2)| (v1 - v2).abs() < 1e-7)
        }

        fn prop_grad(val: Vec<f64>) -> bool {
            let g = Identity.grad(Array1::from(val));

            g.into_iter().all(|v| (v - 1.0).abs() < 1e-7)
        }

        quickcheck(prop_transform as fn(Vec<f64>) -> bool);
        quickcheck(prop_grad as fn(Vec<f64>) -> bool);
    }

    #[test]
    fn test_softplus_f64() {
        assert!((Softplus.transform(0.0) - 0.693147).abs() < 1e-5);
        assert!((Softplus.transform(1.0) - 1.31326).abs() < 1e-5);
        assert!((Softplus.transform(2.0) - 2.12693).abs() < 1e-5);

        assert!((Softplus.grad(0.0) - Logistic::default().transform(0.0)).abs() < 1e-7);
        assert!((Softplus.grad(1.0) - Logistic::default().transform(1.0)).abs() < 1e-7);
        assert!((Softplus.grad(2.0) - Logistic::default().transform(2.0)).abs() < 1e-7);
    }

    #[test]
    fn test_softplus_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Softplus.transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }

    #[test]
    fn test_logistic_f64() {
        let l = Logistic::standard();

        assert!((l.transform(0.0) - 0.5).abs() < 1e-7);
        assert!((l.transform(1.0) - 1.0 / (1.0 + 1.0 / E)).abs() < 1e-7);
        assert!((l.transform(2.0) - 1.0 / (1.0 + 1.0 / E / E)).abs() < 1e-7);

        assert!((l.grad(0.0) - 0.25).abs() < 1e-5);
        assert!((l.grad(1.0) - 0.196612).abs() < 1e-5);
        assert!((l.grad(2.0) - 0.104994).abs() < 1e-5);
    }

    #[test]
    fn test_logistic_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Logistic::default().transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }

    #[test]
    fn test_exponential_f64() {
        assert!((Exp.transform(0.0) - 1.0).abs() < 1e-7);
        assert!((Exp.transform(1.0) - E).abs() < 1e-7);
        assert!((Exp.transform(2.0) - E * E).abs() < 1e-7);

        assert!((Exp.transform(0.0) - Exp.grad(0.0)).abs() < 1e-7);
        assert!((Exp.transform(1.0) - Exp.grad(1.0)).abs() < 1e-7);
        assert!((Exp.transform(2.0) - Exp.grad(2.0)).abs() < 1e-7);
    }

    #[test]
    fn test_exponential_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Exp.transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}

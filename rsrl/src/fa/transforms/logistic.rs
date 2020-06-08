use ndarray::Array1;
use super::Transform;

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

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use std::f64::consts::E;
    use super::{Logistic, Transform};

    #[test]
    fn test_f64() {
        let l = Logistic::standard();

        assert!((l.transform(0.0) - 0.5).abs() < 1e-7);
        assert!((l.transform(1.0) - 1.0 / (1.0 + 1.0 / E)).abs() < 1e-7);
        assert!((l.transform(2.0) - 1.0 / (1.0 + 1.0 / E / E)).abs() < 1e-7);

        assert!((l.grad(0.0) - 0.25).abs() < 1e-5);
        assert!((l.grad(1.0) - 0.196612).abs() < 1e-5);
        assert!((l.grad(2.0) - 0.104994).abs() < 1e-5);
    }

    #[test]
    fn test_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Logistic::default().transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}

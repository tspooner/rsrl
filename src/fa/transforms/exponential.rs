use crate::geometry::Vector;
use super::Transform;

// f(x) ≜ exp(x)
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
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

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use std::f64::consts::E;
    use super::{Exp, Transform};

    #[test]
    fn test_f64() {
        assert!((Exp.transform(0.0) - 1.0).abs() < 1e-7);
        assert!((Exp.transform(1.0) - E).abs() < 1e-7);
        assert!((Exp.transform(2.0) - E * E).abs() < 1e-7);

        assert!((Exp.transform(0.0) - Exp.grad(0.0)).abs() < 1e-7);
        assert!((Exp.transform(1.0) - Exp.grad(1.0)).abs() < 1e-7);
        assert!((Exp.transform(2.0) - Exp.grad(2.0)).abs() < 1e-7);
    }

    #[test]
    fn test_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Exp.transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}

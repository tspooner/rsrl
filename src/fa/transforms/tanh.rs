use crate::geometry::Vector;
use super::Transform;

// f(x) ≜ tanh(x)
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
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

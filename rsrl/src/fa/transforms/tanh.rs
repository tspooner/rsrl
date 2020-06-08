use ndarray::Array1;
use super::Transform;

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

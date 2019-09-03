use crate::{
    core::{Parameter, Algorithm},
    fa::gradients::{Gradient, PartialDerivative},
    geometry::{Vector, Matrix, MatrixViewMut},
};
use ndarray::{ArrayBase, Data, Ix1};
use std::ops::{AddAssign, MulAssign, Deref, DerefMut};
use super::Trace;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Accumulating<G: Gradient>(G);

impl<G: Gradient> Accumulating<G> {
    pub fn new(grad: G) -> Self { Accumulating(grad) }
}

impl<G: Gradient> Algorithm for Accumulating<G> {
    fn handle_terminal(&mut self) {}
}

impl<G: Gradient> Trace<G> for Accumulating<G> {
    fn update(&mut self, grad: &G) {
        self.0.combine_inplace(grad, |x, y| x + y);
    }

    fn scaled_update(&mut self, factor: f64, grad: &G) {
        self.0.combine_inplace(grad, |x, y| factor * x + y);
    }
}

impl<G: Gradient> Deref for Accumulating<G> {
    type Target = G;

    fn deref(&self) -> &G { &self.0 }
}

impl<G: Gradient> DerefMut for Accumulating<G> {
    fn deref_mut(&mut self) -> &mut G { &mut self.0 }
}

#[cfg(test)]
mod tests {
    use super::{Trace, Accumulating, Matrix};
    use ndarray::arr1;

    const LAMBDA: f64 = 0.95;

    #[test]
    fn test_accumulating() {
        let mut trace = Accumulating::new(Matrix::zeros((10, 1)));

        assert_eq!(trace.0.column(0), arr1(&[0.0f64; 10]));

        trace.scale(LAMBDA);
        assert_eq!(trace.0.column(0), arr1(&[0.0f64; 10]));

        trace.update(&Matrix::ones((10, 1)));
        assert_eq!(trace.0.column(0), arr1(&[1.0f64; 10]));

        trace.scale(LAMBDA);
        assert_eq!(trace.0.column(0), arr1(&[0.95f64; 10]));

        trace.update(&Matrix::ones((10, 1)));
        assert_eq!(trace.0.column(0), arr1(&[1.95f64; 10]));

        trace.reset();
        assert_eq!(trace.0.column(0), arr1(&[0f64; 10]));
    }
}

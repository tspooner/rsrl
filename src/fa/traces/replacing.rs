use crate::{
    core::{Parameter, Algorithm},
    geometry::{Vector, Matrix, MatrixViewMut},
    linalg::MatrixLike,
};
use ndarray::{ArrayBase, Data, Ix1};
use std::ops::{AddAssign, MulAssign, Deref, DerefMut};
use super::Trace;

#[derive(Clone, Debug)]
pub struct Replacing<G: MatrixLike>(G);

impl<G: MatrixLike> Replacing<G> {
    pub fn new(grad: G) -> Self { Replacing(grad) }

    pub fn zeros(dim: [usize; 2]) -> Self { Replacing::new(G::zeros(dim)) }
}

impl<G: MatrixLike> Algorithm for Replacing<G> {
    fn handle_terminal(&mut self) {}
}

impl<G: MatrixLike> Trace<G> for Replacing<G> {
    fn update(&mut self, grad: &G) {
        self.0.combine_inplace(grad, |x, y| f64::max(-1.0, f64::min(1.0, x + y)));
    }

    fn scaled_update(&mut self, factor: f64, grad: &G) {
        self.0.combine_inplace(grad, |x, y| f64::max(-1.0, f64::min(1.0, factor * x + y)));
    }
}

impl<G: MatrixLike> Deref for Replacing<G> {
    type Target = G;

    fn deref(&self) -> &G { &self.0 }
}

impl<G: MatrixLike> DerefMut for Replacing<G> {
    fn deref_mut(&mut self) -> &mut G { &mut self.0 }
}

#[cfg(test)]
mod tests {
    use super::{Trace, Replacing, Matrix};
    use ndarray::arr1;

    const LAMBDA: f64 = 0.95;

    #[test]
    fn test_replacing() {
        let mut trace = Replacing::new(Matrix::zeros((10, 1)));

        assert_eq!(trace.0.column(0), arr1(&[0.0f64; 10]));

        trace.scale(LAMBDA);
        assert_eq!(trace.0.column(0), arr1(&[0.0f64; 10]));

        trace.update(&Matrix::ones((10, 1)));
        assert_eq!(trace.0.column(0), arr1(&[1.0f64; 10]));

        trace.scale(LAMBDA);
        assert_eq!(trace.0.column(0), arr1(&[0.95f64; 10]));

        trace.update(&Matrix::ones((10, 1)));
        assert_eq!(trace.0.column(0), arr1(&[1.0f64; 10]));

        trace.reset();
        assert_eq!(trace.0.column(0), arr1(&[0f64; 10]));
    }
}

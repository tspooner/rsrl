use crate::{Parameter, Algorithm, linalg::MatrixLike};
use std::ops::{Deref, DerefMut};
use super::Trace;

#[derive(Clone, Debug)]
pub struct Dutch<G: MatrixLike> {
    alpha: Parameter,
    grad: G,
}

impl<G: MatrixLike> Dutch<G> {
    pub fn new<T: Into<Parameter>>(alpha: T, grad: G) -> Self {
        Dutch { alpha: alpha.into(), grad }
    }

    pub fn zeros<T: Into<Parameter>>(alpha: T, dim: [usize; 2]) -> Self {
        Dutch::new(alpha, G::zeros(dim))
    }
}

impl<G: MatrixLike> Algorithm for Dutch<G> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
    }
}

impl<G: MatrixLike> Trace<G> for Dutch<G> {
    fn update(&mut self, grad: &G) {
        let scale = 1.0 - self.alpha.value();

        self.grad.combine_inplace(grad, move |x, y| scale * x + y);
    }

    fn scaled_update(&mut self, factor: f64, grad: &G) {
        let scale = factor * (1.0 - self.alpha.value());

        self.grad.combine_inplace(grad, move |x, y| scale * x + y);
    }
}

impl<G: MatrixLike> Deref for Dutch<G> {
    type Target = G;

    fn deref(&self) -> &G { &self.grad }
}

impl<G: MatrixLike> DerefMut for Dutch<G> {
    fn deref_mut(&mut self) -> &mut G { &mut self.grad }
}

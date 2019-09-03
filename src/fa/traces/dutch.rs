use crate::{
    core::{Parameter, Algorithm},
    fa::gradients::{Gradient, PartialDerivative},
    geometry::{Vector, Matrix, MatrixViewMut},
};
use ndarray::{ArrayBase, Data, Ix1};
use std::ops::{AddAssign, MulAssign, Deref, DerefMut};
use super::Trace;

#[derive(Clone, Debug)]
pub struct Dutch<G: Gradient> {
    alpha: Parameter,
    grad: G,
}

impl<G: Gradient> Dutch<G> {
    pub fn new<T: Into<Parameter>>(alpha: T, grad: G) -> Self {
        Dutch { alpha: alpha.into(), grad }
    }
}

impl<G: Gradient> Algorithm for Dutch<G> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
    }
}

impl<G: Gradient> Trace<G> for Dutch<G> {
    fn update(&mut self, grad: &G) {
        let scale = 1.0 - self.alpha.value();

        self.grad.combine_inplace(grad, move |x, y| scale * x + y);
    }

    fn scaled_update(&mut self, factor: f64, grad: &G) {
        let scale = factor * (1.0 - self.alpha.value());

        self.grad.combine_inplace(grad, move |x, y| scale * x + y);
    }
}

impl<G: Gradient> Deref for Dutch<G> {
    type Target = G;

    fn deref(&self) -> &G { &self.grad }
}

impl<G: Gradient> DerefMut for Dutch<G> {
    fn deref_mut(&mut self) -> &mut G { &mut self.grad }
}

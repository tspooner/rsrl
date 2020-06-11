use super::Trace;
use crate::params::{BufferMut, Parameterised};
use ndarray::Dimension;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub struct Dutch<J: BufferMut> {
    alpha: f64,
    buffer: J,
}

impl<J: BufferMut> Dutch<J> {
    pub fn new(alpha: f64, buffer: J) -> Self {
        Dutch {
            alpha: alpha.into(),
            buffer,
        }
    }

    pub fn zeros(alpha: f64, dim: <J::Dim as Dimension>::Pattern) -> Self {
        Dutch::new(alpha, J::zeros(dim))
    }
}

impl<J: BufferMut<Dim = ndarray::Ix2>> Dutch<J> {
    pub fn for_fa<F: Parameterised>(fa: &F, alpha: f64) -> Self {
        Self::zeros(alpha, fa.weights_dim())
    }
}

impl<J: BufferMut> Trace for Dutch<J> {
    type Buffer = J;

    fn update(&mut self, buffer: &J) {
        let scale = 1.0 - self.alpha;

        self.buffer.merge_inplace(buffer, move |x, y| scale * x + y);
    }

    fn scaled_update(&mut self, factor: f64, buffer: &J) {
        let scale = factor * (1.0 - self.alpha);

        self.buffer.merge_inplace(buffer, move |x, y| scale * x + y);
    }
}

impl<J: BufferMut> Deref for Dutch<J> {
    type Target = J;

    fn deref(&self) -> &J { &self.buffer }
}

impl<J: BufferMut> DerefMut for Dutch<J> {
    fn deref_mut(&mut self) -> &mut J { &mut self.buffer }
}

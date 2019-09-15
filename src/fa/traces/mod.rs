//! Eligibility trace types
use crate::core::Algorithm;
use std::ops::{Deref, DerefMut};

pub trait Trace<G: crate::linalg::MatrixLike>: Algorithm + Deref<Target = G> + DerefMut {
    fn update(&mut self, grad: &G);

    fn scale(&mut self, factor: f64) { self.map_inplace(|g| g * factor); }

    fn scaled_update(&mut self, factor: f64, grad: &G) {
        self.scale(factor);
        self.update(grad);
    }

    fn reset(&mut self) { self.scale(0.0) }
}

import_all!(accumulating);
import_all!(dutch);
import_all!(replacing);

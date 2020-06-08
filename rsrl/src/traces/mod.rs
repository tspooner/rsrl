//! Eligibility trace types.
use crate::params::BufferMut;
use std::ops::{Deref, DerefMut};

pub trait Trace: Deref<Target = <Self as Trace>::Buffer> + DerefMut {
    type Buffer: BufferMut;

    fn update(&mut self, buffer: &Self::Buffer);

    fn scale(&mut self, factor: f64) { self.map_inplace(|g| g * factor); }

    fn scaled_update(&mut self, factor: f64, buffer: &Self::Buffer) {
        self.scale(factor);
        self.update(buffer);
    }

    fn reset(&mut self) { self.scale(0.0) }
}

import_all!(accumulating);
import_all!(dutch);
import_all!(replacing);

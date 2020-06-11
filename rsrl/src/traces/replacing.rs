use super::Trace;
use crate::params::BufferMut;
use ndarray::Dimension;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub struct Replacing<B: BufferMut>(B);

impl<B: BufferMut> Replacing<B> {
    pub fn new(buffer: B) -> Self { Replacing(buffer) }

    pub fn zeros(dim: <B::Dim as Dimension>::Pattern) -> Self { Replacing::new(B::zeros(dim)) }
}

impl<B: BufferMut> Trace for Replacing<B> {
    type Buffer = B;

    fn update(&mut self, buffer: &B) {
        self.0
            .merge_inplace(buffer, |x, y| f64::max(-1.0, f64::min(1.0, x + y)));
    }

    fn scaled_update(&mut self, factor: f64, buffer: &B) {
        self.0
            .merge_inplace(buffer, |x, y| f64::max(-1.0, f64::min(1.0, factor * x + y)));
    }
}

impl<B: BufferMut> Deref for Replacing<B> {
    type Target = B;

    fn deref(&self) -> &B { &self.0 }
}

impl<B: BufferMut> DerefMut for Replacing<B> {
    fn deref_mut(&mut self) -> &mut B { &mut self.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array2};

    const LAMBDA: f64 = 0.95;

    #[test]
    fn test_replacing() {
        let mut trace = Replacing::new(Array2::zeros((10, 1)));

        assert_eq!(trace.0.column(0), arr1(&[0.0f64; 10]));

        trace.scale(LAMBDA);
        assert_eq!(trace.0.column(0), arr1(&[0.0f64; 10]));

        trace.update(&Array2::ones((10, 1)));
        assert_eq!(trace.0.column(0), arr1(&[1.0f64; 10]));

        trace.scale(LAMBDA);
        assert_eq!(trace.0.column(0), arr1(&[0.95f64; 10]));

        trace.update(&Array2::ones((10, 1)));
        assert_eq!(trace.0.column(0), arr1(&[1.0f64; 10]));

        trace.reset();
        assert_eq!(trace.0.column(0), arr1(&[0f64; 10]));
    }
}

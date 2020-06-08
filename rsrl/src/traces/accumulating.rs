use crate::params::{Buffer, BufferMut};
use ndarray::Dimension;
use std::ops::{Deref, DerefMut};
use super::Trace;

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Accumulating<B: Buffer>(B);

impl<B: BufferMut> Accumulating<B> {
    pub fn new(buffer: B) -> Self { Accumulating(buffer) }

    pub fn zeros(dim: <B::Dim as Dimension>::Pattern) -> Self {
        Accumulating::new(B::zeros(dim))
    }
}

impl<B: BufferMut> Trace for Accumulating<B> {
    type Buffer = B;

    fn update(&mut self, buffer: &B) {
        self.0.merge_inplace(buffer, |x, y| x + y);
    }

    fn scaled_update(&mut self, alpha: f64, buffer: &B) {
        self.0.merge_inplace(buffer, |x, y| alpha * x + y);
    }
}

impl<B: Buffer> Deref for Accumulating<B> {
    type Target = B;

    fn deref(&self) -> &B { &self.0 }
}

impl<B: Buffer> DerefMut for Accumulating<B> {
    fn deref_mut(&mut self) -> &mut B { &mut self.0 }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, Array2};
    use super::*;

    const LAMBDA: f64 = 0.95;

    #[test]
    fn test_accumulating() {
        let mut trace = Accumulating::new(Array2::zeros((10, 1)));

        assert_eq!(trace.0.column(0), arr1(&[0.0f64; 10]));

        trace.scale(LAMBDA);
        assert_eq!(trace.0.column(0), arr1(&[0.0f64; 10]));

        trace.update(&Array2::ones((10, 1)));
        assert_eq!(trace.0.column(0), arr1(&[1.0f64; 10]));

        trace.scale(LAMBDA);
        assert_eq!(trace.0.column(0), arr1(&[0.95f64; 10]));

        trace.update(&Array2::ones((10, 1)));
        assert_eq!(trace.0.column(0), arr1(&[1.95f64; 10]));

        trace.reset();
        assert_eq!(trace.0.column(0), arr1(&[0f64; 10]));
    }
}

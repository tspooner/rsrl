use crate::{
    core::DerefSlice,
    fa::{StateFunction, StateActionFunction, FiniteActionFunction, Parameterised},
    geometry::Matrix,
};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use std::ops::IndexMut;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct IndexedTabular(Vec<Vec<f64>>);

impl IndexedTabular {
    pub fn new(weights: Vec<Vec<f64>>) -> Self {
        IndexedTabular(weights)
    }

    pub fn zeros(dim: [usize; 2]) -> Self {
        IndexedTabular(vec![vec![0.0; dim[0]]; dim[1]])
    }
}

// Q(s, a):
impl StateActionFunction<usize, usize> for IndexedTabular {
    type Output = f64;

    fn evaluate(&self, state: &usize, action: &usize) -> f64 {
        self.0[*action][*state]
    }

    fn update(&mut self, state: &usize, action: &usize, error: f64) {
        *self.0.index_mut(*action).index_mut(*state) += error;
    }
}

impl FiniteActionFunction<usize> for IndexedTabular {
    fn n_actions(&self) -> usize { self.0.len() }

    fn evaluate_all(&self, state: &usize) -> Vec<f64> {
        self.0.iter().map(|c| c[*state]).collect()
    }

    fn update_all(&mut self, state: &usize, errors: Vec<f64>) {
        for (c, e) in self.0.iter_mut().zip(errors.into_iter()) {
            *c.index_mut(*state) += e;
        }
    }
}

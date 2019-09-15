use crate::fa::{StateActionFunction, EnumerableStateActionFunction};
use std::ops::IndexMut;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Tabular(Vec<Vec<f64>>);

impl Tabular {
    pub fn new(weights: Vec<Vec<f64>>) -> Self {
        Tabular(weights)
    }

    pub fn zeros(dim: [usize; 2]) -> Self {
        Tabular(vec![vec![0.0; dim[0]]; dim[1]])
    }
}

// Q(s, a):
impl StateActionFunction<usize, usize> for Tabular {
    type Output = f64;

    fn evaluate(&self, state: &usize, action: &usize) -> f64 {
        self.0[*action][*state]
    }

    fn update(&mut self, state: &usize, action: &usize, error: f64) {
        *self.0.index_mut(*action).index_mut(*state) += error;
    }
}

impl EnumerableStateActionFunction<usize> for Tabular {
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

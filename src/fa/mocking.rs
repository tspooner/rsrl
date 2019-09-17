use crate::{
    Shared, make_shared,
    fa::{StateActionFunction, EnumerableStateActionFunction, Parameterised},
    geometry::{MatrixView, MatrixViewMut},
};
use std::marker::PhantomData;

pub struct MockQ {
    output: Option<Vec<f64>>,
}

impl MockQ {
    pub fn new(output: Option<Vec<f64>>) -> Self { MockQ { output: output } }

    pub fn new_shared(output: Option<Vec<f64>>) -> Shared<Self> {
        make_shared(Self::new(output))
    }

    #[allow(dead_code)]
    pub fn set_output(&mut self, output: Vec<f64>) { self.output = Some(output) }

    #[allow(dead_code)]
    pub fn clear_output(&mut self) { self.output = None }
}

impl Parameterised for MockQ {
    fn weights_view(&self) -> MatrixView<f64> { unimplemented!() }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { unimplemented!() }
}

impl StateActionFunction<Vec<f64>, usize> for MockQ {
    type Output = f64;

    fn evaluate(&self, x: &Vec<f64>, action: &usize) -> f64 {
        match self.output {
            Some(ref out) => out[*action].clone(),
            None => x[*action].clone(),
        }
    }

    fn update(&mut self, _: &Vec<f64>, _: &usize, _: f64) {}
}

impl EnumerableStateActionFunction<Vec<f64>> for MockQ {
    fn n_actions(&self) -> usize {
        match self.output {
            Some(ref out) => out.len(),
            None => unimplemented!(),
        }
    }

    fn evaluate_all(&self, x: &Vec<f64>) -> Vec<f64> {
        match self.output {
            Some(ref out) => out.clone(),
            None => x.clone(),
        }
    }

    fn update_all(&mut self, _: &Vec<f64>, _: Vec<f64>) {}
}

use core::{make_shared, Shared};
use fa::{Approximator, EvaluationResult, QFunction, UpdateResult, VFunction};
use geometry::Vector;
use std::marker::PhantomData;

pub struct MockQ {
    output: Option<Vector<f64>>,
}

impl MockQ {
    pub fn new(output: Option<Vector<f64>>) -> Self { MockQ { output: output } }

    pub fn new_shared(output: Option<Vector<f64>>) -> Shared<Self> {
        make_shared(Self::new(output))
    }

    pub fn set_output(&mut self, output: Vector<f64>) { self.output = Some(output) }

    pub fn clear_output(&mut self) { self.output = None }
}

impl Approximator<Vec<f64>> for MockQ {
    type Value = Vector<f64>;

    fn evaluate(&self, s: &Vec<f64>) -> EvaluationResult<Vector<f64>> {
        Ok(match self.output {
            Some(ref vector) => vector.clone(),
            None => s.clone().into(),
        })
    }

    fn update(&mut self, p: &Vec<f64>, errors: Vector<f64>) -> UpdateResult<()> { Ok(()) }
}

impl QFunction<Vec<f64>> for MockQ {
    fn n_actions(&self) -> usize {
        match self.output {
            Some(ref vector) => vector.len(),
            None => 0,
        }
    }
}

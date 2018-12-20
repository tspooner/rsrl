use crate::core::{make_shared, Shared};
use crate::fa::{Approximator, EvaluationResult, QFunction, UpdateResult, VFunction};
use crate::geometry::Vector;
use std::marker::PhantomData;

pub struct MockQ {
    output: Option<Vector<f64>>,
}

impl MockQ {
    pub fn new(output: Option<Vector<f64>>) -> Self { MockQ { output: output } }

    pub fn new_shared(output: Option<Vector<f64>>) -> Shared<Self> {
        make_shared(Self::new(output))
    }

    #[allow(dead_code)]
    pub fn set_output(&mut self, output: Vector<f64>) { self.output = Some(output) }

    #[allow(dead_code)]
    pub fn clear_output(&mut self) { self.output = None }
}

impl Approximator<Vector<f64>> for MockQ {
    type Value = Vector<f64>;

    fn evaluate(&self, s: &Vector<f64>) -> EvaluationResult<Vector<f64>> {
        Ok(match self.output {
            Some(ref vector) => vector.clone(),
            None => s.clone().into(),
        })
    }

    fn update(&mut self, _: &Vector<f64>, _: Vector<f64>) -> UpdateResult<()> { Ok(()) }
}

impl QFunction<Vector<f64>> for MockQ {
    fn n_actions(&self) -> usize {
        match self.output {
            Some(ref vector) => vector.len(),
            None => 0,
        }
    }
}

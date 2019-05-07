use crate::{
    core::{make_shared, Shared},
    fa::{
        Approximator,
        Embedding,
        EvaluationResult,
        Features,
        Parameterised,
        QFunction,
        UpdateResult,
    },
    geometry::{Matrix, MatrixView, MatrixViewMut, Vector},
};
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

impl Embedding<Vector<f64>> for MockQ {
    fn n_features(&self) -> usize { unimplemented!() }

    fn embed(&self, s: &Vector<f64>) -> Features { Features::Dense(s.clone()) }
}

impl Parameterised for MockQ {
    fn weights_view(&self) -> MatrixView<f64> { unimplemented!() }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { unimplemented!() }
}

impl Approximator for MockQ {
    type Output = Vector<f64>;

    fn n_outputs(&self) -> usize {
        match self.output {
            Some(ref out) => out.len(),
            None => 0,
        }
    }

    fn evaluate(&self, f: &Features) -> EvaluationResult<Vector<f64>> {
        Ok(match self.output {
            Some(ref out) => out.clone(),
            None => match f {
                Features::Sparse(_) => unimplemented!(),
                Features::Dense(ref out) => out.clone(),
            },
        })
    }

    fn jacobian(&self, _: &Features) -> Matrix<f64> { unimplemented!() }

    fn update_grad(&mut self, _: &Matrix<f64>, _: Vector<f64>) -> UpdateResult<()> { Ok(()) }

    fn update(&mut self, _: &Features, _: Vector<f64>) -> UpdateResult<()> { Ok(()) }
}

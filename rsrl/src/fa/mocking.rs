use crate::{core::*, make_shared, params::*, Shared};

pub struct MockQ {
    output: Option<Vec<f64>>,
}

impl MockQ {
    pub fn new(output: Option<Vec<f64>>) -> Self { MockQ { output: output } }

    pub fn new_shared(output: Option<Vec<f64>>) -> Shared<Self> { make_shared(Self::new(output)) }

    #[allow(dead_code)]
    pub fn set_output(&mut self, output: Vec<f64>) { self.output = Some(output) }

    #[allow(dead_code)]
    pub fn clear_output(&mut self) { self.output = None }
}

impl Parameterised for MockQ {
    fn weights_view(&self) -> WeightsView { unimplemented!() }

    fn weights_view_mut(&mut self) -> WeightsViewMut { unimplemented!() }
}

impl<S: std::borrow::Borrow<Vec<f64>>> Function<(S,)> for MockQ {
    type Output = Vec<f64>;

    fn evaluate(&self, (x,): (S,)) -> Vec<f64> {
        match self.output {
            Some(ref out) => out.clone(),
            None => x.borrow().clone(),
        }
    }
}

impl<S, A> Function<(S, A)> for MockQ
where
    S: std::borrow::Borrow<Vec<f64>>,
    A: std::borrow::Borrow<usize>,
{
    type Output = f64;

    fn evaluate(&self, (x, action): (S, A)) -> f64 {
        match self.output {
            Some(ref out) => out[*action.borrow()].clone(),
            None => x.borrow()[*action.borrow()].clone(),
        }
    }
}

impl<S: std::borrow::Borrow<Vec<f64>>> Enumerable<(S,)> for MockQ {
    // fn n_outputs(&self) -> usize {
    // match self.output {
    // Some(ref out) => out.len(),
    // None => unimplemented!(),
    // }
    // }

    fn evaluate_index(&self, (x,): (S,), index: usize) -> f64 {
        match self.output {
            Some(ref out) => out[index],
            None => x.borrow()[index],
        }
    }
}

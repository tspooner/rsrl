use crate::{
    core::*,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised, StateFunction},
};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

#[derive(Parameterised)]
pub struct GradientMC<V> {
    #[weights] pub v_func: V,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<V> GradientMC<V> {
    pub fn new<T1, T2>(v_func: V, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        GradientMC {
            v_func,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<V> Algorithm for GradientMC<V> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, V> BatchLearner<S, A> for GradientMC<V>
where
    V: StateFunction<S, Output = f64>
{
    fn handle_batch(&mut self, batch: &[Transition<S, A>]) {
        let mut sum = 0.0;

        batch.into_iter().rev().for_each(|ref t| {
            sum = t.reward + self.gamma * sum;

            let s = t.from.state();
            let v = self.v_func.evaluate(s);

            self.v_func.update(s, sum - v);
        })
    }
}

impl<S, V> ValuePredictor<S> for GradientMC<V>
where
    V: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 {
        self.v_func.evaluate(s)
    }
}

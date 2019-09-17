use crate::{
    BatchLearner,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised, StateFunction},
    prediction::ValuePredictor,
};

#[derive(Parameterised)]
pub struct GradientMC<V> {
    #[weights] pub v_func: V,

    pub alpha: f64,
    pub gamma: f64,
}

impl<V> GradientMC<V> {
    pub fn new(v_func: V, alpha: f64, gamma: f64) -> Self {
        GradientMC {
            v_func,

            alpha,
            gamma,
        }
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

use crate::{
    OnlineLearner,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised, StateFunction},
    prediction::ValuePredictor,
};

#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
pub struct TD<V> {
    #[weights] pub v_func: V,

    pub alpha: f64,
    pub gamma: f64,
}

impl<V> TD<V> {
    pub fn new(v_func: V, alpha: f64, gamma: f64) -> Self {
        TD {
            v_func,

            alpha,
            gamma,
        }
    }
}

impl<S, A, V> OnlineLearner<S, A> for TD<V>
where
    V: StateFunction<S, Output = f64>
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let s = t.from.state();
        let v = self.v_func.evaluate(s);

        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.v_func.evaluate(t.to.state()) - v
        };

        self.v_func.update(s, self.alpha * td_error);
    }
}

impl<S, V> ValuePredictor<S> for TD<V>
where
    V: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 { self.v_func.evaluate(s) }
}

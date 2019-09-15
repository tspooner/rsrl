use crate::{
    core::*,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised, StateFunction},
};

#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
pub struct TD<V> {
    #[weights] pub v_func: V,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<V> TD<V> {
    pub fn new<T1, T2>(v_func: V, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TD {
            v_func,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<V> Algorithm for TD<V> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
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

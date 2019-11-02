use crate::{
    Algorithm, OnlineLearner, Parameter,
    domains::Transition,
    fa::{Parameterised, StateFunction},
    geometry::{Matrix, MatrixView, MatrixViewMut},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarianceTD<J, V> {
    pub value_estimator: J,
    pub variance_estimator: V,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<J, V> VarianceTD<J, V> {
    pub fn new<T1, T2>(value_estimator: J, variance_estimator: V, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        VarianceTD {
            value_estimator,
            variance_estimator,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<J, V> VarianceTD<J, V> {
    fn compute_value_error<S>(&self, s: &S, reward: f64, ns: &S) -> f64
    where
        J: StateFunction<S, Output = f64>,
    {
        reward + self.value_estimator.evaluate(ns) - self.value_estimator.evaluate(s)
    }
}

impl<J, V> Algorithm for VarianceTD<J, V> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, J, V> OnlineLearner<S, A> for VarianceTD<J, V>
where
    J: StateFunction<S, Output = f64>,
    V: StateFunction<S, Output = f64>,
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let (s, ns) = t.states();

        let value_error = self.compute_value_error(s, t.reward, ns);
        let meta_reward = value_error * value_error;
        let variance_est = self.variance_estimator.evaluate(s);

        let td_error = if t.terminated() {
            meta_reward - variance_est
        } else {
            let gamma_var = self.gamma * self.gamma;

            meta_reward + gamma_var * self.variance_estimator.evaluate(ns) - variance_est
        };

        self.variance_estimator.update(s, self.alpha * td_error);
    }
}

impl<J, V: Parameterised> Parameterised for VarianceTD<J, V> {
    fn weights(&self) -> Matrix<f64> {
        self.variance_estimator.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.variance_estimator.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.variance_estimator.weights_view_mut()
    }
}

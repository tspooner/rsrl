use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Parameterised, Approximator, VFunction};
use crate::geometry::{Matrix, MatrixView, MatrixViewMut};

pub struct TD<V> {
    pub v_func: V,

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

impl<S, A, V: VFunction<S>> OnlineLearner<S, A> for TD<V> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let phi_s = self.v_func.to_features(t.from.state());
        let v = self.v_func.evaluate(&phi_s).unwrap();

        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.predict_v(t.to.state()) - v
        };

        self.v_func.update(&phi_s, self.alpha * td_error).ok();
    }
}

impl<S, V: VFunction<S>> ValuePredictor<S> for TD<V> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.v_func.evaluate(&self.v_func.to_features(s)).unwrap()
    }
}

impl<S, A, V: VFunction<S>> ActionValuePredictor<S, A> for TD<V> {}

impl<V: Parameterised> Parameterised for TD<V> {
    fn weights(&self) -> Matrix<f64> {
        self.v_func.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.v_func.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.v_func.weights_view_mut()
    }
}

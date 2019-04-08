use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, Features, Projector, VFunction};
use crate::geometry::{Space, MatrixView, MatrixViewMut};

pub struct GTD2<F> {
    pub fa_theta: F,
    pub fa_w: F,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<F: Parameterised> GTD2<F> {
    pub fn new<T1, T2, T3>(
        fa_theta: F,
        fa_w: F,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        if fa_theta.weights_dim() != fa_w.weights_dim() {
            panic!("fa_theta and fa_w must be equivalent function approximators.")
        }

        GTD2 {
            fa_theta,
            fa_w,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<F> Algorithm for GTD2<F> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F: VFunction<S>> OnlineLearner<S, A> for GTD2<F> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let (phi_s, phi_ns) = t.map_states(|s| self.fa_theta.to_features(s));

        let v = self.fa_theta.evaluate(&phi_s).unwrap();

        let td_estimate = self.fa_w.evaluate(&phi_s).unwrap();
        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.fa_theta.evaluate(&phi_ns).unwrap() - v
        };

        self.fa_w.update(&phi_s, self.beta * (td_error - td_estimate)).unwrap();

        let dim = self.fa_theta.n_features();
        let pd = phi_s.expanded(dim) - self.gamma.value() * phi_ns.expanded(dim);

        self.fa_theta.update(&Features::Dense(pd), self.alpha * td_estimate).ok();
    }
}

impl<S, F: VFunction<S>> ValuePredictor<S> for GTD2<F> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.to_features(s)).unwrap()
    }
}

impl<S, A, F: VFunction<S>> ActionValuePredictor<S, A> for GTD2<F> {}

impl<F: Parameterised> Parameterised for GTD2<F> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa_theta.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.fa_theta.weights_view_mut()
    }
}

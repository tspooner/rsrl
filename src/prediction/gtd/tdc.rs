use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, Features, Projector, VFunction};
use crate::geometry::{Space, MatrixView, MatrixViewMut};

pub struct TDC<F> {
    pub fa_theta: Shared<F>,
    pub fa_w: Shared<F>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<F: Parameterised> TDC<F> {
    pub fn new<T1, T2, T3>(
        fa_theta: Shared<F>,
        fa_w: Shared<F>,
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

        TDC {
            fa_theta,
            fa_w,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<F> Algorithm for TDC<F> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F: VFunction<S>> OnlineLearner<S, A> for TDC<F> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let (phi_s, phi_ns) = t.map_states(|s| self.fa_theta.to_features(s));

        let v = self.fa_theta.evaluate(&phi_s).unwrap();

        let td_estimate = self.fa_w.evaluate(&phi_s).unwrap();
        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.fa_theta.evaluate(&phi_ns).unwrap() - v
        };

        self.fa_w.borrow_mut().update(&phi_s, self.beta * (td_error - td_estimate)).ok();

        let dim = self.fa_theta.n_features();
        let phi =
            td_error * phi_s.expanded(dim) -
            td_estimate * self.gamma.value() * phi_ns.expanded(dim);

        self.fa_theta.borrow_mut().update(&Features::Dense(phi), self.alpha.value()).ok();
    }
}

impl<S, F: VFunction<S>> ValuePredictor<S> for TDC<F> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.to_features(s)).unwrap()
    }
}

impl<S, A, F: VFunction<S>> ActionValuePredictor<S, A> for TDC<F> {}

impl<F: Parameterised> Parameterised for TDC<F> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa_theta.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }
}

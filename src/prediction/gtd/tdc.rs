use core::{Algorithm, Predictor, Parameter};
use domains::Transition;
use fa::{Approximator, Parameterised, Projection, Projector, SimpleLFA, VFunction};
use geometry::Matrix;

pub struct TDC<S: ?Sized, P: Projector<S>> {
    pub fa_theta: SimpleLFA<S, P>,
    pub fa_w: SimpleLFA<S, P>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S: ?Sized, P: Projector<S>> TDC<S, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: SimpleLFA<S, P>,
        fa_w: SimpleLFA<S, P>,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        if fa_theta.projector.dim() != fa_w.projector.dim() {
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

    #[inline(always)]
    fn update_theta(&mut self, phi_s: Projection, phi_ns: Projection,
                    td_error: f64, td_estimate: f64)
    {
        let dim = self.fa_theta.projector.dim();
        let phi =
            td_error * phi_s.expanded(dim) -
            td_estimate * self.gamma.value() * phi_ns.expanded(dim);

        self.fa_theta.update_phi(&Projection::Dense(phi), self.alpha.value());
    }

    #[inline(always)]
    fn update_w(&mut self, phi_s: &Projection, error: f64) {
        self.fa_w.update_phi(phi_s, self.beta * error);
    }
}

impl<S, A, M: Projector<S>> Algorithm<S, A> for TDC<S, M> {
    fn handle_sample(&mut self, t: &Transition<S, A>) {
        let phi_s = self.fa_theta.projector.project(&t.from.state());
        let phi_ns = self.fa_theta.projector.project(&t.to.state());

        let td_error = t.reward + self.gamma * self.fa_theta.evaluate_phi(&phi_ns)
            - self.fa_theta.evaluate_phi(&phi_s);
        let td_estimate = self.fa_w.evaluate_phi(&phi_s);

        self.update_w(&phi_s, td_error - td_estimate);
        self.update_theta(phi_s, phi_ns, td_error, td_estimate);
    }

    fn handle_terminal(&mut self, t: &Transition<S, A>) {
        {
            let phi_s = self.fa_theta.projector.project(&t.from.state());
            let phi_ns = self.fa_theta.projector.project(&t.to.state());

            let td_error = t.reward - self.fa_theta.evaluate_phi(&phi_s);
            let td_estimate = self.fa_w.evaluate_phi(&phi_s);

            self.update_w(&phi_s, td_error - td_estimate);
            self.update_theta(phi_s, phi_ns, td_error, td_estimate);
        }

        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, P: Projector<S>> Predictor<S, A> for TDC<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 { self.fa_theta.evaluate(s).unwrap() }
}

impl<S, P: Projector<S>> Parameterised for TDC<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }
}

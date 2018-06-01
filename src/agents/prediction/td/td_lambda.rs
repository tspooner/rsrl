use agents::{Predictor, TDPredictor, memory::Trace};
use domains::Transition;
use fa::{Approximator, Projection, Projector, SimpleLFA, VFunction};
use {Handler, Parameter};

pub struct TDLambda<S: ?Sized, P: Projector<S>> {
    trace: Trace,

    pub fa_theta: SimpleLFA<S, P>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<S: ?Sized, P: Projector<S>> TDLambda<S, P> {
    pub fn new<T1, T2>(trace: Trace, fa_theta: SimpleLFA<S, P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TDLambda {
            trace: trace,

            fa_theta: fa_theta,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }

    fn compute_error_with_phi(&self, phi_s: &Projection, phi_ns: &Projection, reward: f64) -> f64 {
        let v: f64 = self.fa_theta.evaluate_phi(phi_s);
        let nv: f64 = self.fa_theta.evaluate_phi(phi_ns);

        reward + self.gamma * nv - v
    }

    fn handle_error_with_phi(&mut self, phi_s: Projection, error: f64) {
        let rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(rate);
        self.trace
            .update(&phi_s.expanded(self.fa_theta.projector.dim()));

        self.fa_theta
            .update_phi(&Projection::Dense(self.trace.get()), self.alpha * error);
    }
}

impl<S, P: Projector<S>> Handler<Transition<S, ()>> for TDLambda<S, P> {
    fn handle_sample(&mut self, sample: &Transition<S, ()>) {
        let phi_s = self.fa_theta.projector.project(&sample.from.state());
        let phi_ns = self.fa_theta.projector.project(&sample.to.state());

        let td_error = self.compute_error_with_phi(&phi_s, &phi_ns, sample.reward);

        self.handle_error_with_phi(phi_s, td_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, ()>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
    }
}

impl<S, P: Projector<S>> Predictor<S> for TDLambda<S, P> {
    fn predict(&mut self, s: &S) -> f64 { self.fa_theta.evaluate(s).unwrap() }
}

impl<S, P: Projector<S>> TDPredictor<S> for TDLambda<S, P> {
    fn handle_td_error(&mut self, sample: &Transition<S, ()>, error: f64) {
        let phi_s = self.fa_theta.projector.project(&sample.from.state());

        self.handle_error_with_phi(phi_s, error);
    }

    fn compute_td_error(&self, sample: &Transition<S, ()>) -> f64 {
        let v: f64 = self.fa_theta.evaluate(&sample.from.state()).unwrap();
        let nv: f64 = self.fa_theta.evaluate(&sample.to.state()).unwrap();

        sample.reward + self.gamma * nv - v
    }
}

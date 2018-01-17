use Parameter;
use agents::{Agent, LinearAgent, Predictor, TDPredictor};
use agents::memory::Trace;
use fa::{Function, VFunction, Projector, Projection, Linear};
use geometry::Space;
use ndarray::Array2;

use std::marker::PhantomData;


pub struct TD<S: Space, V: VFunction<S>> {
    pub v_func: V,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, V: VFunction<S>> TD<S, V> {
    pub fn new<T1, T2>(v_func: V, alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        TD {
            v_func: v_func,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, V: VFunction<S>> Agent<S> for TD<S, V> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let td_error = self.compute_error(sample);

        self.handle_error(&sample, td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S>> LinearAgent<S> for TD<S, Linear<S, P>> {
    fn weights(&self) -> Array2<f64> {
        self.v_func.weights.clone()
    }
}

impl<S: Space, V: VFunction<S>> Predictor<S> for TD<S, V> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.v_func.evaluate(s)
    }
}

impl<S: Space, V: VFunction<S>> TDPredictor<S> for TD<S, V> {
    fn compute_error(&self, sample: &Self::Sample) -> f64 {
        let v = self.v_func.evaluate(&sample.0);
        let nv = self.v_func.evaluate(&sample.1);

        sample.2 + self.gamma*nv - v
    }

    fn handle_error(&mut self, sample: &Self::Sample, error: f64) {
        self.v_func.update(&sample.0, self.alpha*error);
    }
}


pub struct TDLambda<S: Space, P: Projector<S>> {
    trace: Trace,

    pub fa_theta: Linear<S, P>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<S: Space, P: Projector<S>> TDLambda<S, P> {
    pub fn new<T1, T2>(trace: Trace, fa_theta: Linear<S, P>, alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
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

        reward + self.gamma*nv - v
    }

    fn handle_error_with_phi(&mut self, phi_s: Projection, td_error: f64) {
        let rate = self.trace.lambda.value()*self.gamma.value();

        self.trace.decay(rate);
        self.trace.update(&self.fa_theta.projector.expand_projection(phi_s));

        self.fa_theta.update_phi(&Projection::Dense(self.trace.get()), self.alpha*td_error);
    }
}

impl<S: Space, P: Projector<S>> Agent<S> for TDLambda<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        let td_error = self.compute_error_with_phi(&phi_s, &phi_ns, sample.2);

        self.handle_error_with_phi(phi_s, td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
    }
}

impl<S: Space, P: Projector<S>> LinearAgent<S> for TDLambda<S, P> {
    fn weights(&self) -> Array2<f64> {
        self.fa_theta.weights.clone()
    }
}

impl<S: Space, P: Projector<S>> Predictor<S> for TDLambda<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.fa_theta.evaluate(s)
    }
}

impl<S: Space, P: Projector<S>> TDPredictor<S> for TDLambda<S, P> {
    fn compute_error(&self, sample: &Self::Sample) -> f64 {
        let phi_s = self.fa_theta.projector.project(&sample.0);
        let phi_ns = self.fa_theta.projector.project(&sample.1);

        self.compute_error_with_phi(&phi_s, &phi_ns, sample.2)
    }

    fn handle_error(&mut self, sample: &Self::Sample, td_error: f64) {
        let phi_s = self.fa_theta.projector.project(&sample.0);

        self.handle_error_with_phi(phi_s, td_error);
    }
}


// TODO:
// ETD(lambda) - https://arxiv.org/pdf/1503.04269.pdf
// HTD(lambda) - https://arxiv.org/pdf/1602.08771.pdf
// PTD(lambda) - http://proceedings.mlr.press/v32/sutton14.pdf
// True online TD(lambda) - http://proceedings.mlr.press/v32/seijen14.pdf
// True online ETD(lambda) - https://arxiv.org/pdf/1602.08771.pdf
// True online ETD(beta, lambda) - https://arxiv.org/pdf/1602.08771.pdf
// True online HTD(lambda) - https://arxiv.org/pdf/1602.08771.pdf

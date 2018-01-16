use Parameter;
use agents::PredictionAgent;
use agents::memory::Trace;
use fa::{Function, VFunction, Projector, Projection, Linear};
use geometry::Space;

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

impl<S: Space, V: VFunction<S>> PredictionAgent<S> for TD<S, V> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.v_func.evaluate(s)
    }

    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let v = self.v_func.evaluate(s);
        let nv = self.v_func.evaluate(ns);

        let td_error = r + self.gamma*nv - v;
        self.v_func.update(&s, self.alpha*td_error);

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
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
}

impl<S: Space, P: Projector<S>> PredictionAgent<S> for TDLambda<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.v_func.evaluate(s)
    }

    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.fa_theta.projector.project(s);
        let phi_ns = self.fa_theta.projector.project(ns);

        let rate = self.trace.lambda.value()*self.gamma.value();
        let td_error = r + self.gamma*self.fa_theta.evaluate_phi(&phi_ns) -
            self.fa_theta.evaluate_phi(&phi_s);

        self.trace.decay(rate);
        self.trace.update(&self.fa_theta.projector.expand_projection(phi_s));

        self.fa_theta.update_phi(&Projection::Dense(self.trace.get()), self.alpha*td_error);

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
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

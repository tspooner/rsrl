use Parameter;
use agents::PredictionAgent;
use agents::memory::Trace;
use fa::{VFunction, Projection, Linear};
use geometry::Space;

use std::marker::PhantomData;


pub struct TD<S: Space, V: VFunction<S>> {
    v_func: V,

    alpha: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, V> TD<S, V>
    where V: VFunction<S>
{
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

impl<S: Space, V> PredictionAgent<S> for TD<S, V>
    where V: VFunction<S>
{
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let v = self.v_func.evaluate(s);
        let nv = self.v_func.evaluate(ns);

        let td_error = r + self.gamma * nv - v;
        self.v_func.update(&s, self.alpha * td_error);

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}


pub struct TDLambda<S: Space, P: Projection<S>> {
    trace: Trace,
    v_func: Linear<S, P>,

    alpha: Parameter,
    gamma: Parameter,
}

impl<S: Space, P: Projection<S>> TDLambda<S, P> {
    pub fn new<T1, T2>(trace: Trace, v_func: Linear<S, P>, alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        TDLambda {
            trace: trace,
            v_func: v_func,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<S: Space, P: Projection<S>> PredictionAgent<S> for TDLambda<S, P> {
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.v_func.project(s);
        let phi_ns = self.v_func.project(ns);

        self.trace.decay(self.gamma.value());
        self.trace.update(&phi_s);

        let td_error = r + self.gamma * self.v_func.evaluate_phi(&phi_ns) -
                       self.v_func.evaluate_phi(&phi_s);

        self.v_func.update_phi(self.trace.get(), self.alpha * td_error);

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
    }
}

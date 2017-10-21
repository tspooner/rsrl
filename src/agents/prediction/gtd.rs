use Parameter;
use fa::{VFunction, Projection, Linear};
use agents::PredictionAgent;
use geometry::Space;


pub struct GTD2<S: Space, P: Projection<S>>
{
    v_func: Linear<S, P>,
    a_func: Linear<S, P>,

    alpha: Parameter,
    beta: Parameter,
    gamma: Parameter,
}

impl<S: Space, P: Projection<S>> GTD2<S, P> {
    pub fn new<T1, T2, T3>(v_func: Linear<S, P>, a_func: Linear<S, P>,
                           alpha: T1, beta: T2, gamma: T3) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
    {
        if !v_func.equivalent(&a_func) {
            panic!("v_func and a_func must be equivalent function approximators.")
        }

        GTD2 {
            v_func: v_func,
            a_func: a_func,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<S: Space, V> PredictionAgent<S> for GTD2<S, V>
    where V: VFunction<S> + Projection<S>
{
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.v_func.project(s);
        let phi_ns = self.v_func.project(ns);

        let td_error = r + self.gamma*self.v_func.evaluate_phi(&phi_ns) -
            self.v_func.evaluate_phi(&phi_s);
        let td_estimate = self.a_func.evaluate_phi(&phi_s);

        self.v_func.update_phi(&(&phi_s - &(self.gamma.value()*&phi_ns)), self.alpha*td_estimate);
        self.a_func.update_phi(&phi_s, self.beta*(td_error - td_estimate));

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}


pub struct TDC<S: Space, P: Projection<S>>
{
    v_func: Linear<S, P>,
    a_func: Linear<S, P>,

    alpha: Parameter,
    beta: Parameter,
    gamma: Parameter,
}

impl<S: Space, P: Projection<S>> TDC<S, P> {
    pub fn new<T1, T2, T3>(v_func: Linear<S, P>, a_func: Linear<S, P>,
                           alpha: T1, beta: T2, gamma: T3) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
    {
        if !v_func.equivalent(&a_func) {
            panic!("v_func and a_func must be equivalent function approximators.")
        }

        TDC {
            v_func: v_func,
            a_func: a_func,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<S: Space, V> PredictionAgent<S> for TDC<S, V>
    where V: VFunction<S> + Projection<S>
{
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.v_func.project(s);
        let phi_ns = self.v_func.project(ns);

        let td_error = r + self.gamma*self.v_func.evaluate_phi(&phi_ns) -
            self.v_func.evaluate_phi(&phi_s);
        let td_estimate = self.a_func.evaluate_phi(&phi_s);

        self.v_func.update_phi(&(&(td_error*&phi_s) - &(self.gamma.value()*td_estimate*&phi_ns)), self.alpha.value());
        self.a_func.update_phi(&phi_s, self.beta*(td_error - td_estimate));

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}


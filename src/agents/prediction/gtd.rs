use Parameter;
use fa::{VFunction, Linear};
use agents::PredictionAgent;
use geometry::Space;
use std::marker::PhantomData;


pub struct GTD2<S: Space, V: VFunction<S> + Linear<S>>
{
    v_func: V,
    a_func: V,

    alpha: Parameter,
    beta: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, V> GTD2<S, V>
    where V: VFunction<S> + Linear<S>
{
    pub fn new<T1, T2, T3>(v_func: V, a_func: V,
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

            phantom: PhantomData,
        }
    }
}

impl<S: Space, V> PredictionAgent<S> for GTD2<S, V>
    where V: VFunction<S> + Linear<S>
{
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.v_func.phi(s);
        let phi_ns = self.v_func.phi(ns);

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


pub struct TDC<S: Space, V: VFunction<S> + Linear<S>>
{
    v_func: V,
    a_func: V,

    alpha: Parameter,
    beta: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, V> TDC<S, V>
    where V: VFunction<S> + Linear<S>
{
    pub fn new<T1, T2, T3>(v_func: V, a_func: V,
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

            phantom: PhantomData,
        }
    }
}

impl<S: Space, V> PredictionAgent<S> for TDC<S, V>
    where V: VFunction<S> + Linear<S>
{
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.v_func.phi(s);
        let phi_ns = self.v_func.phi(ns);

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


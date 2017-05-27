use Parameter;
use fa::{VFunction, Linear};
use agents::PredictionAgent;
use geometry::{Space, NullSpace};
use std::marker::PhantomData;

use ndarray::ArrayBase;


pub struct TD<S: Space, V: VFunction<S>>
{
    v_func: V,

    alpha: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, V: VFunction<S>> TD<S, V>
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

impl<S: Space, V: VFunction<S>> PredictionAgent<S> for TD<S, V>
{
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


pub struct GTD2<S: Space, V: VFunction<S> + Linear<S>>
{
    v_func: V,
    a_func: V,

    alpha: Parameter,
    beta: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, V: VFunction<S> + Linear<S>> GTD2<S, V>
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

impl<S: Space, V: VFunction<S> + Linear<S>> PredictionAgent<S> for GTD2<S, V>
{
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64> {
        let phi_s = self.v_func.phi(s);
        let phi_ns = self.v_func.phi(ns);

        let phi_diff = &phi_s - &(self.gamma.value()*&phi_ns);

        let td_error = r + self.gamma*self.v_func.evaluate_phi(&phi_ns) -
            self.v_func.evaluate_phi(&phi_s);
        let td_estimate = self.a_func.evaluate_phi(&phi_s);

        self.v_func.update_phi(&phi_diff, self.alpha*td_estimate);
        self.a_func.update_phi(&phi_s, self.beta*(td_error - td_estimate));

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

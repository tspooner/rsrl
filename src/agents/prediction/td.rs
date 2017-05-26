use Parameter;
use fa::VFunction;
use agents::PredictionAgent;
use geometry::{Space, NullSpace};
use std::marker::PhantomData;


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

        let td_error = self.alpha*(r + self.gamma*nv - v);
        self.v_func.update(&s, td_error);

        Some(td_error)
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

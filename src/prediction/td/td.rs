use core::{Algorithm, Predictor, Parameter};
use domains::Transition;
use fa::{Parameterised, VFunction};
use geometry::Matrix;
use std::marker::PhantomData;

pub struct TD<S: ?Sized, V: VFunction<S>> {
    pub v_func: V,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: ?Sized, V: VFunction<S>> TD<S, V> {
    pub fn new<T1, T2>(v_func: V, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TD {
            v_func,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }

    #[inline(always)]
    fn update_v(&mut self, s: &S, error: f64) {
        let _ = self.v_func.update(s, self.alpha * error);
    }
}

impl<S, A, V: VFunction<S>> Algorithm<S, A> for TD<S, V>
where
    Self: Predictor<S, A>
{
    fn handle_sample(&mut self, t: &Transition<S, A>) {
        let s = t.from.state();
        let v = self.predict_v(&s);
        let nv = self.predict_v(&t.to.state());

        let td_error = t.reward + self.gamma * nv - v;

        self.update_v(&s, td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, A>) {
        let s = t.from.state();
        let td_error = t.reward - self.predict_v(&t.from.state());

        self.update_v(&s, td_error);

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, V: VFunction<S>> Predictor<S, A> for TD<S, V> {
    fn predict_v(&mut self, s: &S) -> f64 { self.v_func.evaluate(s).unwrap() }
}

impl<S, V: VFunction<S> + Parameterised> Parameterised for TD<S, V> {
    fn weights(&self) -> Matrix<f64> {
        self.v_func.weights()
    }
}

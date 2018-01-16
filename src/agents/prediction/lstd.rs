use Parameter;
use fa::{Function, Projector, Linear};
use utils:: argmaxima;
use agents::{Agent, BatchAgent, Predictor};
use agents::memory::Trace;
use ndarray::{Array1, Array2};
use geometry::Space;
use ndarray_linalg::Solve;
use std::marker::PhantomData;


// TODO: Add LSTDLambda, iLSTD and iLSTDLambda implementations based on the approach described in
// "Incremental Least-Squares Temporal Difference Learning" (2006).
// TODO: Implement regularized LSTD "http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2012b.pdf

pub struct LSTD<S: Space, P: Projector<S>>
{
    a: Array2<f64>,
    b: Array1<f64>,

    beta: Linear<S, P>,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Projector<S>> LSTD<S, P> {
    pub fn new<T: Into<Parameter>>(beta: Linear<S, P>, gamma: T) -> Self {
        let n_features = beta.projector.dim();

        LSTD {
            a: Array2::zeros((n_features, n_features)),
            b: Array1::zeros((n_features,)),

            beta: beta,
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Projector<S>> Agent<S> for LSTD<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.beta.projector.project_expanded(&sample.0);
        let phi_ns = self.beta.projector.project_expanded(&sample.1);

        let pd = &phi_s - &(self.gamma.value()*phi_ns);

        self.a += &phi_s.broadcast((1, phi_s.dim())).unwrap().dot(&pd);
        self.b.scaled_add(sample.2, &phi_s);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.consolidate();

        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S>> BatchAgent<S> for LSTD<S, P> {
    fn consolidate(&mut self) {
        let d = (self.b.dim(), 1);

        self.beta.assign(self.a.solve_into(self.b.clone()).unwrap().into_shape(d).unwrap());
    }
}

impl<S: Space, P: Projector<S>> Predictor<S> for LSTD<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.beta.evaluate(s)
    }
}


pub struct LSTDLambda<S: Space, P: Projector<S>>
{
    trace: Trace,

    a: Array2<f64>,
    b: Array1<f64>,

    beta: Linear<S, P>,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Projector<S>> LSTDLambda<S, P> {
    pub fn new<T: Into<Parameter>>(trace: Trace, beta: Linear<S, P>, gamma: T) -> Self {
        let n_features = beta.projector.dim();

        LSTDLambda {
            trace: trace,

            a: Array2::zeros((n_features, n_features)),
            b: Array1::zeros((n_features,)),

            beta: beta,
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Projector<S>> Agent<S> for LSTDLambda<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.beta.projector.project_expanded(&sample.0);
        let phi_ns = self.beta.projector.project_expanded(&sample.1);

        let pd = &phi_s - &(self.gamma.value()*&phi_ns);

        self.a += self.trace.get().dot(&pd);
        self.b.scaled_add(sample.2, &phi_s);

        self.trace.decay(self.gamma.value());
        self.trace.update(&phi_ns);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.consolidate();

        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S>> BatchAgent<S> for LSTDLambda<S, P> {
    fn consolidate(&mut self) {
        let d = (self.b.dim(), 1);

        self.beta.assign(self.a.solve_into(self.b.clone()).unwrap().into_shape(d).unwrap());
    }
}

impl<S: Space, P: Projector<S>> Predictor<S> for LSTDLambda<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.beta.evaluate(s)
    }
}


#[allow(non_camel_case_types)]
pub struct iLSTD<S: Space, P: Projector<S>>
{
    fa: Linear<S, P>,
    n_updates: usize,

    a: Array2<f64>,
    mu: Array1<f64>,

    alpha: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Projector<S>> iLSTD<S, P> {
    pub fn new<T1, T2>(fa: Linear<S, P>, n_updates: usize, alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        let n_features = fa.projector.dim();

        iLSTD {
            fa: fa,
            n_updates: n_updates,

            a: Array2::zeros((n_features, n_features)),
            mu: Array1::zeros((n_features,)),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Projector<S>> Agent<S> for iLSTD<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        let phi_s = self.fa.projector.project_expanded(&sample.0);
        let phi_ns = self.fa.projector.project_expanded(&sample.1);

        let da = &phi_s*&(&phi_s - &(self.gamma.value()*phi_ns));
        let db = sample.2*phi_s;

        self.a.zip_mut_with(&da, |y, &x| *y = *y + x);
        self.mu.zip_mut_with(&(db - &da*&self.fa.weights), |y, &x| *y = *y + x);

        let alpha_t = self.alpha.value();

        for i in 0..self.n_updates {
            let (_, idx) = argmaxima(self.mu.mapv(|v| v.abs()).as_slice().unwrap());
            let update = alpha_t*self.mu[idx[0]];

            for j in idx {
                self.fa.weights[(j, 0)] += update;
            }

            // TODO: Improve this, if we have collisions we can probably do something better than
            //       just using the first idx for the mu update (maybe even just a mean).
            self.mu.scaled_add(-update, &self.a.column(i));
        }
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S>> Predictor<S> for iLSTD<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.fa.evaluate(s)
    }
}
use Parameter;
use fa::{Function, VFunction, Projector, Projection, Linear};
use agents::{Agent, BatchAgent, Predictor};
use ndarray::{Array1, Array2};
use geometry::Space;
use ndarray_linalg::Solve;
use std::marker::PhantomData;


pub struct LambdaLSPE<S: Space, P: Projector<S>> {
    a: Array2<f64>,
    b: Array1<f64>,

    last_state: Option<S::Repr>,
    trajectory: Vec<(S::Repr, f64)>,

    pub fa: Linear<S, P>,
    pub alpha: Parameter,
    pub gamma: Parameter,
    pub lambda: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Projector<S>> LambdaLSPE<S, P> {
    pub fn new<T1, T2, T3>(fa: Linear<S, P>, alpha: T1, gamma: T2, lambda: T3) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>,
    {
        let n_features = fa.projector.dim();

        LambdaLSPE {
            a: Array2::zeros((n_features, n_features)),
            b: Array1::zeros((n_features,)),

            last_state: None,
            trajectory: vec![],

            fa: fa,
            alpha: alpha.into(),
            gamma: gamma.into(),
            lambda: lambda.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Projector<S>> Agent<S> for LambdaLSPE<S, P> {
    type Sample = (S::Repr, S::Repr, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        self.last_state = Some(sample.1.clone());
        self.trajectory.push((sample.0.clone(), sample.2));
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.consolidate();

        self.gamma = self.gamma.step();
    }
}

impl<S: Space, P: Projector<S>> BatchAgent<S> for LambdaLSPE<S, P> {
    fn consolidate(&mut self) {
        let mut error = 0.0;

        while let Some(ref ns) = self.last_state {
            if let Some((s, r)) = self.trajectory.pop() {
                let phi_s = self.fa.projector.project(&s);

                let v: f64 = self.fa.evaluate_phi(&phi_s);
                let nv: f64 = self.fa.evaluate(ns);

                error = self.gamma*self.lambda*error + (r + self.gamma*nv - v);

                let phi_s = self.fa.projector.expand_projection(phi_s);

                self.b.scaled_add(v + error, &phi_s);
                self.a += phi_s.dot(&phi_s);
            } else {
                break;
            }
        }

        let estimate = self.a.solve_into(self.b.clone()).unwrap();
        let update = estimate - self.fa.weights.column(0);

        self.fa.update_phi(&Projection::Dense(update), self.alpha.value());

        self.last_state = None;
        self.trajectory.clear();
    }
}

impl<S: Space, P: Projector<S>> Predictor<S> for LambdaLSPE<S, P> {
    fn evaluate(&self, s: &S::Repr) -> f64 {
        self.fa.evaluate(s)
    }
}

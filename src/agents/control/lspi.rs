use Parameter;
use agents::{Agent, BatchAgent, LinearAgent, Controller};
use agents::memory::Trace;
use domains::Transition;
use fa::{Function, QFunction, Projector, Projection, Linear};
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use utils::dot;
use std::marker::PhantomData;


pub struct LSPILambda<S: Space, M: Projector<S>, P: Policy> {
    trace: Trace,

    a: Array2<f64>,
    b: Array1<f64>,

    last_state: Option<S::Repr>,
    trajectory: Vec<(S::Repr, f64)>,

    pub fa: Linear<S, M>,
    pub policy: P,

    pub gamma: Parameter,
    pub lambda: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, M: Projector<S>, P: Policy> LSPILambda<S, M, P> {
    pub fn new<T1, T2>(trace: Trace, fa: Linear<S, M>, policy: P, gamma: T1, lambda: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
    {
        let n_features = fa.projector.dim();

        LSPILambda {
            trace: trace,

            a: Array2::zeros((n_features, n_features)),
            b: Array1::zeros((n_features,)),

            last_state: None,
            trajectory: vec![],

            fa: fa,
            policy: policy,

            gamma: gamma.into(),
            lambda: lambda.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, M: Projector<S>, P: Policy> Agent<S> for LSPILambda<S, M, P> {
    type Sample = Transition<S, ActionSpace>;

    fn handle_sample(&mut self, sample: &Self::Sample) {
        self.last_state = Some(sample.to.state().clone());
        self.trajectory.push((sample.from.state().clone(), sample.reward));
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.consolidate();

        self.gamma = self.gamma.step();
        self.lambda = self.lambda.step();
    }
}

impl<S: Space, M: Projector<S>, P: Policy> BatchAgent<S> for LSPILambda<S, M, P> {
    fn consolidate(&mut self) {
        while let Some(ref ns) = self.last_state {
            if let Some((s, r)) = self.trajectory.pop() {
                let nqs: Vec<f64> = self.fa.evaluate(ns);
                let exp_nqs = dot(&nqs, &Greedy.probabilities(nqs.as_slice()));

                let phi_s = self.fa.projector.project_expanded(&s);

                self.trace.decay(self.gamma*self.lambda);
                self.trace.update(&phi_s);

                let z = self.trace.get();

                self.a += z.dot(&(phi_s.clone() - self.gamma*exp_nqs));
                self.b.scaled_add(r, &phi_s);
            } else {
                break;
            }
        }

        let d = (self.b.dim(), 1);
        self.fa.assign(self.a.solve_into(self.b.clone()).unwrap().into_shape(d).unwrap());

        self.last_state = None;
        self.trajectory.clear();
    }
}

impl<S: Space, M: Projector<S>, P: Policy> LinearAgent<S> for LSPILambda<S, M, P> {
    fn weights(&self) -> Array2<f64> {
        self.fa.weights.clone()
    }
}

impl<S: Space, M: Projector<S>, P: Policy> Controller<S, ActionSpace> for LSPILambda<S, M, P> {
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.evaluate_policy(&mut Greedy, s)
    }

    fn mu(&mut self, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.fa.evaluate(s);

        self.policy.sample(qs.as_slice())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.fa.evaluate(s);

        p.sample(qs.as_slice())
    }
}
